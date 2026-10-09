//! Whisper-compatible frontend: periodic Hann, reflected STFT, Slaney mel scale.
use crate::CancellationToken;
use anyhow::Result;
use candle_core::{Device, Tensor};
use rustfft::{FftPlanner, num_complex::Complex};

pub(super) fn mel(samples: &[f32], cancellation: &CancellationToken) -> Result<Tensor> {
    cancellation.check()?;
    const FFT: usize = 400;
    const HOP: usize = 160;
    const BINS: usize = 128;
    // Short inputs need enough samples for PyTorch's reflected center padding.
    let length = samples.len().max(FFT).div_ceil(HOP) * HOP;
    let mut signal = samples.to_vec();
    signal.resize(length, 0.0);
    let frames = length / HOP;
    let frequencies = FFT / 2 + 1;
    let filters = filters(BINS, frequencies);
    let window: Vec<_> = (0..FFT)
        .map(|i| 0.5 - 0.5 * (2.0 * std::f32::consts::PI * i as f32 / FFT as f32).cos())
        .collect();
    let fft = FftPlanner::<f32>::new().plan_fft_forward(FFT);
    let mut scratch = vec![Complex::default(); fft.get_inplace_scratch_len()];
    let mut buffer = vec![Complex::default(); FFT];
    let mut values = vec![0.0f32; BINS * frames];
    for frame in 0..frames {
        cancellation.check()?;
        for (i, b) in buffer.iter_mut().enumerate() {
            let index = frame as isize * HOP as isize + i as isize - (FFT / 2) as isize;
            let reflected = if index < 0 {
                -index
            } else if index >= length as isize {
                2 * length as isize - 2 - index
            } else {
                index
            };
            *b = Complex::new(signal[reflected as usize] * window[i], 0.0);
        }
        fft.process_with_scratch(&mut buffer, &mut scratch);
        // The spectrum is no longer needed after its power is computed. Reuse
        // its real slots instead of recomputing power for every mel channel.
        for bin in &mut buffer[..frequencies] {
            bin.re = bin.norm_sqr();
        }
        for m in 0..BINS {
            values[m * frames + frame] = (0..frequencies)
                .map(|f| filters[m * frequencies + f] * buffer[f].re)
                .sum::<f32>()
                .max(1e-10)
                .log10();
        }
    }
    let floor = values.iter().copied().fold(f32::NEG_INFINITY, f32::max) - 8.0;
    for v in &mut values {
        *v = (v.max(floor) + 4.0) / 4.0;
    }
    cancellation.check()?;
    Ok(Tensor::from_vec(values, (BINS, frames), &Device::Cpu)?)
}

fn filters(bins: usize, frequencies: usize) -> Vec<f32> {
    let log_step = 6.4f64.ln() / 27.0;
    let mel_max = 15.0 + (8000.0f64 / 1000.0).ln() / log_step;
    let edges: Vec<_> = (0..bins + 2)
        .map(|i| {
            let m = mel_max * i as f64 / (bins + 1) as f64;
            if m < 15.0 {
                m * 200.0 / 3.0
            } else {
                1000.0 * ((m - 15.0) * log_step).exp()
            }
        })
        .collect();
    let mut out = vec![0.0; bins * frequencies];
    for m in 0..bins {
        for f in 0..frequencies {
            let hz = f as f64 * 16000.0 / 400.0;
            let down = (hz - edges[m]) / (edges[m + 1] - edges[m]);
            let up = (edges[m + 2] - hz) / (edges[m + 2] - edges[m + 1]);
            out[m * frequencies + f] =
                (down.min(up).max(0.0) * 2.0 / (edges[m + 2] - edges[m])) as f32;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn silence_and_short_audio_are_finite() {
        for n in [1, 160, 16000] {
            let m = mel(&vec![0.0; n], &CancellationToken::default()).unwrap();
            assert_eq!(m.dims()[0], 128);
            assert!(
                m.flatten_all()
                    .unwrap()
                    .to_vec1::<f32>()
                    .unwrap()
                    .iter()
                    .all(|v| (*v + 1.5).abs() < 1e-5)
            );
        }
    }
}
