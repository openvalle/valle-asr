use anyhow::{Context, Result, ensure};
use std::path::Path;

/// Validated, normalized mono PCM. Model backends receive 16 kHz f32 samples.
#[derive(Debug, Clone)]
pub struct Audio {
    samples: Vec<f32>,
}

impl Audio {
    pub const SAMPLE_RATE: u32 = 16_000;

    pub fn from_mono(samples: Vec<f32>, sample_rate: u32) -> Result<Self> {
        ensure!(sample_rate > 0, "sample rate must be positive");
        ensure!(!samples.is_empty(), "audio is empty");
        ensure!(
            samples
                .iter()
                .all(|v| v.is_finite() && (-1.0..=1.0).contains(v)),
            "PCM samples must be finite and within [-1, 1]"
        );
        // Windowed-sinc low-pass resampling avoids aliasing when downsampling.
        let samples = if sample_rate == Self::SAMPLE_RATE {
            samples
        } else {
            resample(&samples, sample_rate, Self::SAMPLE_RATE)
        };
        Ok(Self { samples })
    }

    pub fn from_wav(path: impl AsRef<Path>) -> Result<Self> {
        let mut wav = hound::WavReader::open(path.as_ref())
            .with_context(|| format!("open WAV {}", path.as_ref().display()))?;
        let spec = wav.spec();
        ensure!(spec.channels > 0, "WAV has no channels");
        let interleaved = match spec.sample_format {
            hound::SampleFormat::Float => wav.samples::<f32>().collect::<Result<Vec<_>, _>>()?,
            hound::SampleFormat::Int => {
                ensure!(
                    (1..=32).contains(&spec.bits_per_sample),
                    "unsupported PCM depth"
                );
                let scale = 2f32.powi(i32::from(spec.bits_per_sample) - 1);
                wav.samples::<i32>()
                    .map(|v| v.map(|x| x as f32 / scale))
                    .collect::<Result<Vec<_>, _>>()?
            }
        };
        let channels = usize::from(spec.channels);
        ensure!(interleaved.len() % channels == 0, "incomplete WAV frame");
        let samples = interleaved
            .chunks_exact(channels)
            .map(|frame| frame.iter().copied().sum::<f32>() / channels as f32)
            .collect();
        Self::from_mono(samples, spec.sample_rate)
    }

    pub fn samples(&self) -> &[f32] {
        &self.samples
    }

    pub fn duration_ms(&self) -> u64 {
        (self.samples.len() as u64 * 1000).div_ceil(u64::from(Self::SAMPLE_RATE))
    }
}

fn resample(input: &[f32], from: u32, to: u32) -> Vec<f32> {
    let count = (input.len() as u64 * u64::from(to)).div_ceil(u64::from(from)) as usize;
    let cutoff = (f64::from(to) / f64::from(from)).min(1.0) * 0.95;
    let radius = (24.0 / cutoff).ceil() as i64;
    (0..count)
        .map(|i| {
            let pos = i as f64 * f64::from(from) / f64::from(to);
            let center = pos.floor() as i64;
            let mut value = 0.0;
            let mut weight = 0.0;
            for j in center - radius..=center + radius {
                let delta = pos - j as f64;
                let x = std::f64::consts::PI * delta * cutoff;
                let sinc = if x.abs() < 1e-12 { 1.0 } else { x.sin() / x };
                let window =
                    0.5 * (1.0 + (std::f64::consts::PI * delta / (radius + 1) as f64).cos());
                let w = sinc * window;
                value += f64::from(input[j.clamp(0, input.len() as i64 - 1) as usize]) * w;
                weight += w;
            }
            (value / weight).clamp(-1.0, 1.0) as f32
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_invalid_audio() {
        assert!(Audio::from_mono(vec![], 16000).is_err());
        assert!(Audio::from_mono(vec![f32::NAN], 16000).is_err());
        assert!(Audio::from_mono(vec![1.1], 16000).is_err());
        assert!(Audio::from_mono(vec![0.0], 0).is_err());
    }

    #[test]
    fn resampling_preserves_duration_and_rejects_aliasing() {
        let signal: Vec<_> = (0..4800)
            .map(|i| (2.0 * std::f32::consts::PI * 12000.0 * i as f32 / 48000.0).sin())
            .collect();
        let audio = Audio::from_mono(signal, 48000).unwrap();
        assert_eq!(audio.duration_ms(), 100);
        let rms = (audio.samples()[100..1500]
            .iter()
            .map(|x| x * x)
            .sum::<f32>()
            / 1400.0)
            .sqrt();
        assert!(rms < 0.01, "aliased high-frequency energy: {rms}");
    }
}
