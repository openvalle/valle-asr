use crate::Audio;
use anyhow::{Context, Result, ensure};
use hound::{Sample, SampleFormat, WavReader, WavSpec};
use std::{
    fs::File,
    io::{BufReader, Read, Seek, SeekFrom},
    path::Path,
};

/// One normalized source interval. Sample offsets are on the 16 kHz timeline.
pub struct AudioChunk {
    pub audio: Audio,
    pub start_sample: u64,
}

impl AudioChunk {
    pub fn start_ms(&self) -> u64 {
        (self.start_sample * 1000).div_ceil(u64::from(Audio::SAMPLE_RATE))
    }

    pub fn end_ms(&self) -> u64 {
        ((self.start_sample + self.audio.samples().len() as u64) * 1000)
            .div_ceil(u64::from(Audio::SAMPLE_RATE))
    }
}

/// A seekable WAV decoder with bounded input/output buffers. No whole-file PCM
/// is retained. Integer/float channels are mixed while decoding, before resampling.
/// Resampling uses absolute positions and real neighboring source samples, so
/// chunk boundaries do not reset its phase or introduce artificial padding.
pub struct WavChunks {
    input: BufReader<File>,
    spec: WavSpec,
    byte_width: u16,
    data_start: u64,
    source_frames: u64,
    total_samples: u64,
    position: u64,
    max_samples: usize,
    kernels: Kernels,
}

impl WavChunks {
    pub fn open(path: impl AsRef<Path>, chunk_seconds: u32) -> Result<Self> {
        ensure!(
            (1..=30).contains(&chunk_seconds),
            "chunk_seconds must be 1..=30"
        );
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
            (source_frames * u64::from(Audio::SAMPLE_RATE)).div_ceil(u64::from(spec.sample_rate));
        Ok(Self {
            input,
            spec,
            byte_width,
            data_start,
            source_frames,
            total_samples,
            position: 0,
            max_samples: chunk_seconds as usize * Audio::SAMPLE_RATE as usize,
            kernels: Kernels::new(spec.sample_rate),
        })
    }

    pub fn duration_ms(&self) -> u64 {
        (self.total_samples * 1000).div_ceil(u64::from(Audio::SAMPLE_RATE))
    }

    /// Reuse the same open file for a second pass (for example word alignment).
    pub fn rewind(&mut self) {
        self.position = 0;
    }

    pub fn next_chunk(&mut self) -> Result<Option<AudioChunk>> {
        if self.position == self.total_samples {
            return Ok(None);
        }
        let count = (self.total_samples - self.position).min(self.max_samples as u64) as usize;
        let target_rate = u64::from(Audio::SAMPLE_RATE);
        let source_rate = u64::from(self.spec.sample_rate);
        let radius = if source_rate == target_rate {
            0
        } else {
            self.kernels.radius as u64
        };
        let first = (self.position * source_rate / target_rate).saturating_sub(radius);
        let last = (((self.position + count as u64 - 1) * source_rate / target_rate) + radius)
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
        let mut samples = if source_rate == target_rate {
            raw
        } else {
            let mut normalized = Vec::with_capacity(count);
            for index in self.position..self.position + count as u64 {
                let numerator = index * source_rate;
                let center = (numerator / target_rate) as i64;
                let phase = (numerator % target_rate) as u32;
                let radius = self.kernels.radius;
                let weights = self.kernels.get(phase);
                let mut sum = 0.0;
                for (tap, weight) in weights.iter().enumerate() {
                    let frame = (center + tap as i64 - radius)
                        .clamp(0, self.source_frames as i64 - 1)
                        as u64;
                    sum += f64::from(raw[(frame - first) as usize]) * weight;
                }
                normalized.push(sum.clamp(-1.0, 1.0) as f32);
            }
            normalized
        };
        if self.position + (count as u64) < self.total_samples {
            samples.truncate(quiet_cut(&samples));
        }
        let start_sample = self.position;
        self.position += samples.len() as u64;
        Ok(Some(AudioChunk {
            audio: Audio::from_mono(samples, Audio::SAMPLE_RATE)?,
            start_sample,
        }))
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
        let cutoff = (f64::from(Audio::SAMPLE_RATE) / f64::from(source_rate)).min(1.0) * 0.95;
        let radius = (24.0 / cutoff).ceil() as i64;
        let mut a = source_rate;
        let mut b = Audio::SAMPLE_RATE;
        while b != 0 {
            (a, b) = (b, a % b);
        }
        let phases = Audio::SAMPLE_RATE / a;
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
        let fraction = f64::from(phase) / f64::from(Audio::SAMPLE_RATE);
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

// Look back at most three seconds (or a quarter of a small chunk) for a 120 ms
// quiet interval. Prefer its midpoint; sustained silence at the hard limit can
// use the entire chunk. Continuous speech falls back to the hard limit.
fn quiet_cut(samples: &[f32]) -> usize {
    let frame = 320;
    let lookback = (3 * Audio::SAMPLE_RATE as usize).min(samples.len() / 4);
    let start = samples.len() - lookback;
    let mut run = None;
    let mut candidate = None;
    for (index, block) in samples[start..].chunks_exact(frame).enumerate() {
        let quiet = block.iter().map(|x| f64::from(*x).powi(2)).sum::<f64>() / frame as f64 <= 1e-5;
        if quiet {
            run.get_or_insert(start + index * frame);
        } else if let Some(begin) = run.take() {
            let end = start + index * frame;
            if end - begin >= 6 * frame {
                candidate = Some((begin + end) / 2);
            }
        }
    }
    if run.is_some_and(|begin| samples.len() - begin >= 6 * frame) {
        samples.len()
    } else {
        candidate.unwrap_or(samples.len())
    }
}
