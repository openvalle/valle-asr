use anyhow::{Context, Result, ensure};
use hound::{Sample, SampleFormat, WavReader, WavSpec};
use std::{
    fs::File,
    io::{BufReader, Read, Seek, SeekFrom},
    path::Path,
};
use valle_asr::{AudioSource, SAMPLE_RATE};

/// A seekable WAV decoder that reads any range of the 16 kHz mono timeline.
/// No whole-file PCM is retained. Integer/float channels are mixed while
/// decoding, before resampling. Resampling uses absolute positions and real
/// neighboring source samples, so read boundaries do not reset its phase or
/// introduce artificial padding.
pub struct Wav {
    input: BufReader<File>,
    spec: WavSpec,
    byte_width: u16,
    data_start: u64,
    source_frames: u64,
    total_samples: u64,
    kernels: Kernels,
}

impl Wav {
    /// Open a RIFF WAV of integer/float PCM up to 384 kHz.
    ///
    /// # Errors
    /// Rejects empty, truncated or unsupported WAV data, and filesystem
    /// failures. RF64 and compressed audio are unsupported.
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let reader = WavReader::open(path.as_ref())
            .with_context(|| format!("open WAV {}", path.as_ref().display()))?;
        let spec = reader.spec();
        ensure!(spec.channels > 0, "WAV has no channels");
        // Bound both source-rate buffers and resampling filter width, including
        // for malformed headers. Standard audio rates up to 384 kHz are supported.
        ensure!(
            (1..=384_000).contains(&spec.sample_rate),
            "WAV sample rate must be 1..=384000 Hz"
        );
        ensure!(
            (1..=32).contains(&spec.bits_per_sample),
            "unsupported PCM depth"
        );
        let source_frames = u64::from(reader.duration());
        let source_values = u64::from(reader.len());
        ensure!(source_frames > 0, "audio is empty");
        let mut input = reader.into_inner();
        let data_start = input.stream_position()?;
        // Hound has validated the RIFF header and leaves its reader at the data
        // payload. Retain the actual container width, including 24 bits in a
        // 32-bit container, rather than deriving seeks from valid sample bits.
        input.seek(SeekFrom::Start(
            data_start.checked_sub(4).context("WAV data header")?,
        ))?;
        let mut bytes = [0; 4];
        input.read_exact(&mut bytes)?;
        let data_bytes = u64::from(u32::from_le_bytes(bytes));
        ensure!(
            data_bytes.is_multiple_of(source_values),
            "incomplete WAV sample"
        );
        let byte_width = u16::try_from(data_bytes / source_values)?;
        ensure!(
            (1..=4).contains(&byte_width),
            "unsupported PCM container width"
        );
        ensure!(
            data_start + data_bytes <= input.get_ref().metadata()?.len(),
            "truncated WAV data"
        );
        let total_samples =
            (source_frames * u64::from(SAMPLE_RATE)).div_ceil(u64::from(spec.sample_rate));
        Ok(Self {
            input,
            spec,
            byte_width,
            data_start,
            source_frames,
            total_samples,
            kernels: Kernels::new(spec.sample_rate),
        })
    }
}

impl AudioSource for Wav {
    fn samples(&self) -> u64 {
        self.total_samples
    }

    fn read(&mut self, start: u64, out: &mut [f32]) -> Result<()> {
        let count = out.len();
        if count == 0 {
            return Ok(());
        }
        ensure!(
            start + count as u64 <= self.total_samples,
            "read past the end of the WAV"
        );
        let target_rate = u64::from(SAMPLE_RATE);
        let source_rate = u64::from(self.spec.sample_rate);
        let radius = if source_rate == target_rate {
            0
        } else {
            self.kernels.radius as u64
        };
        let first = (start * source_rate / target_rate).saturating_sub(radius);
        let last = (((start + count as u64 - 1) * source_rate / target_rate) + radius)
            .min(self.source_frames - 1);
        let offset =
            self.data_start + first * u64::from(self.spec.channels) * u64::from(self.byte_width);
        self.input.seek(SeekFrom::Start(offset))?;
        let mut raw = Vec::with_capacity((last - first + 1) as usize);
        let scale = 2f64.powi(i32::from(self.spec.bits_per_sample) - 1);
        for _ in first..=last {
            let mut sum = 0.0;
            for _ in 0..self.spec.channels {
                let sample = match self.spec.sample_format {
                    SampleFormat::Float => f64::from(f32::read(
                        &mut self.input,
                        SampleFormat::Float,
                        self.byte_width,
                        self.spec.bits_per_sample,
                    )?),
                    SampleFormat::Int => {
                        f64::from(i32::read(
                            &mut self.input,
                            SampleFormat::Int,
                            self.byte_width,
                            self.spec.bits_per_sample,
                        )?) / scale
                    }
                };
                ensure!(
                    sample.is_finite() && (-1.0..=1.0).contains(&sample),
                    "PCM samples must be finite and within [-1, 1]"
                );
                sum += sample;
            }
            raw.push((sum / f64::from(self.spec.channels)) as f32);
        }
        if source_rate == target_rate {
            out.copy_from_slice(&raw);
            return Ok(());
        }
        for (index, value) in (start..).zip(out.iter_mut()) {
            let numerator = index * source_rate;
            let center = (numerator / target_rate) as i64;
            let phase = (numerator % target_rate) as u32;
            let radius = self.kernels.radius;
            let weights = self.kernels.get(phase);
            let mut sum = 0.0;
            for (tap, weight) in weights.iter().enumerate() {
                let frame =
                    (center + tap as i64 - radius).clamp(0, self.source_frames as i64 - 1) as u64;
                sum += f64::from(raw[(frame - first) as usize]) * weight;
            }
            *value = sum.clamp(-1.0, 1.0) as f32;
        }
        Ok(())
    }
}

struct Kernels {
    cutoff: f64,
    radius: i64,
    divisor: u32,
    cached: Vec<Option<Vec<f64>>>,
    scratch: Vec<f64>,
}

impl Kernels {
    fn new(source_rate: u32) -> Self {
        let cutoff = (f64::from(SAMPLE_RATE) / f64::from(source_rate)).min(1.0) * 0.95;
        let radius = (24.0 / cutoff).ceil() as i64;
        let mut a = source_rate;
        let mut b = SAMPLE_RATE;
        while b != 0 {
            (a, b) = (b, a % b);
        }
        let phases = SAMPLE_RATE / a;
        Self {
            cutoff,
            radius,
            divisor: a,
            cached: if phases <= 512 {
                vec![None; phases as usize]
            } else {
                vec![]
            },
            scratch: vec![],
        }
    }

    fn fill(cutoff: f64, radius: i64, phase: u32, weights: &mut Vec<f64>) {
        weights.clear();
        let fraction = f64::from(phase) / f64::from(SAMPLE_RATE);
        let mut total = 0.0;
        for tap in -radius..=radius {
            let delta = fraction - tap as f64;
            let x = std::f64::consts::PI * delta * cutoff;
            let sinc = if x.abs() < 1e-12 { 1.0 } else { x.sin() / x };
            let window = 0.5 * (1.0 + (std::f64::consts::PI * delta / (radius + 1) as f64).cos());
            let weight = sinc * window;
            weights.push(weight);
            total += weight;
        }
        for weight in weights {
            *weight /= total;
        }
    }

    fn get(&mut self, phase: u32) -> &[f64] {
        if self.cached.is_empty() {
            Self::fill(self.cutoff, self.radius, phase, &mut self.scratch);
            &self.scratch
        } else {
            self.cached[(phase / self.divisor) as usize].get_or_insert_with(|| {
                let mut weights = vec![];
                Self::fill(self.cutoff, self.radius, phase, &mut weights);
                weights
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::NamedTempFile;

    fn write(spec: WavSpec, bytes_per_sample: u16, frames: &[i32]) -> Result<NamedTempFile> {
        let file = NamedTempFile::new()?;
        let mut writer = hound::WavWriter::new_with_spec_ex(
            file.reopen()?,
            hound::WavSpecEx {
                spec,
                bytes_per_sample,
            },
        )?;
        for frame in frames {
            for _ in 0..spec.channels {
                writer.write_sample(*frame)?;
            }
        }
        writer.finalize()?;
        Ok(file)
    }

    fn read_all(wav: &mut Wav, step: usize) -> Result<Vec<f32>> {
        let mut samples = vec![0.0; wav.samples() as usize];
        for (index, piece) in samples.chunks_mut(step).enumerate() {
            wav.read((index * step) as u64, piece)?;
        }
        Ok(samples)
    }

    #[test]
    fn resampling_is_continuous_across_reads_and_removes_aliasing() -> Result<()> {
        for rate in [22050, 44100, 48000] {
            let spec = WavSpec {
                channels: 2,
                sample_rate: rate,
                bits_per_sample: 16,
                sample_format: SampleFormat::Int,
            };
            let frames: Vec<_> = (0..rate * 2 + 373)
                .map(|i| (12000.0 * (f64::from(i) * 0.15).sin()) as i32)
                .collect();
            let file = write(spec, 2, &frames)?;
            let mut wav = Wav::open(file.path())?;
            assert_eq!(
                wav.samples(),
                (frames.len() as u64 * 16_000).div_ceil(u64::from(rate))
            );
            let whole = read_all(&mut wav, usize::MAX)?;
            assert_eq!(read_all(&mut wav, 16_000)?, whole, "{rate} Hz seams");
            assert_eq!(read_all(&mut wav, 7_919)?, whole, "{rate} Hz seams");
        }
        // 12 kHz is above the 8 kHz limit of 16 kHz audio and must not fold back.
        let spec = WavSpec {
            channels: 1,
            sample_rate: 48_000,
            bits_per_sample: 16,
            sample_format: SampleFormat::Int,
        };
        let frames: Vec<_> = (0..4800)
            .map(|i| (32000.0 * (std::f64::consts::PI * f64::from(i) / 2.0).sin()) as i32)
            .collect();
        let file = write(spec, 2, &frames)?;
        let samples = read_all(&mut Wav::open(file.path())?, usize::MAX)?;
        let rms = (samples[100..1500].iter().map(|x| x * x).sum::<f32>() / 1400.0).sqrt();
        assert!(rms < 0.01, "aliased high-frequency energy: {rms}");
        Ok(())
    }

    #[test]
    fn padded_pcm_and_float_wav_decode_with_correct_container_seeks() -> Result<()> {
        let spec = WavSpec {
            channels: 1,
            sample_rate: 16_000,
            bits_per_sample: 24,
            sample_format: SampleFormat::Int,
        };
        let frames: Vec<_> = (0..33_000).map(|i| i * 71).collect();
        let file = write(spec, 4, &frames)?;
        let mut wav = Wav::open(file.path())?;
        let expected: Vec<_> = frames
            .iter()
            .map(|v| (f64::from(*v) / f64::from(1 << 23)) as f32)
            .collect();
        assert_eq!(read_all(&mut wav, 16_000)?, expected);
        let mut tail = [0.0; 3];
        wav.read(20_000, &mut tail)?;
        assert_eq!(tail, expected[20_000..20_003]);
        assert!(wav.read(32_999, &mut tail).is_err());

        let float = NamedTempFile::new()?;
        let mut writer = hound::WavWriter::new(
            float.reopen()?,
            WavSpec {
                bits_per_sample: 32,
                sample_format: SampleFormat::Float,
                ..spec
            },
        )?;
        for _ in 0..17_000 {
            writer.write_sample(0.25f32)?;
        }
        writer.finalize()?;
        let samples = read_all(&mut Wav::open(float.path())?, 16_000)?;
        assert!(samples.iter().all(|x| *x == 0.25), "float decode changed");
        Ok(())
    }

    #[test]
    fn truncated_wav_is_rejected() -> Result<()> {
        let spec = WavSpec {
            channels: 1,
            sample_rate: 16_000,
            bits_per_sample: 16,
            sample_format: SampleFormat::Int,
        };
        let file = write(spec, 2, &[1; 32_000])?;
        assert!(Wav::open(file.path()).is_ok());
        file.as_file().set_len(100)?;
        assert!(Wav::open(file.path()).is_err());
        Ok(())
    }
}
