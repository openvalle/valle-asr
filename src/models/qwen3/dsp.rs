//! Whisper-compatible frontend: periodic Hann, reflected STFT, Slaney mel scale.
use crate::{CancellationToken, Result};
use candle_core::{Device, Tensor};
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
    let window: Vec<_> = (0..FFT)
        .map(|i| 0.5 - 0.5 * (2.0 * std::f32::consts::PI * i as f32 / FFT as f32).cos())
        .collect();
    let mut windowed = Vec::with_capacity(frames * FFT);
    for frame in 0..frames {
        cancellation.check()?;
        windowed.extend(window.iter().enumerate().map(|(i, w)| {
            let index = frame as isize * HOP as isize + i as isize - (FFT / 2) as isize;
            let reflected = if index < 0 {
                -index
            } else if index >= length as isize {
                2 * length as isize - 2 - index
            } else {
                index
            };
            signal[reflected as usize] * w
        }));
    }
    // A 400-point real DFT as one matrix product: each frame against the
    // cosine and sine of every frequency bin. It runs on the same matrix
    // kernels as the network, so no FFT library is needed.
    let mut basis = vec![0.0f32; FFT * 2 * frequencies];
    for n in 0..FFT {
        for k in 0..frequencies {
            let phase = 2.0 * std::f64::consts::PI * ((n * k) % FFT) as f64 / FFT as f64;
            basis[n * 2 * frequencies + k] = phase.cos() as f32;
            basis[n * 2 * frequencies + frequencies + k] = phase.sin() as f32;
        }
    }
    let spectrum = Tensor::from_vec(windowed, (frames, FFT), &Device::Cpu)?.matmul(
        &Tensor::from_vec(basis, (FFT, 2 * frequencies), &Device::Cpu)?,
    )?;
    cancellation.check()?;
    let power = (spectrum.narrow(1, 0, frequencies)?.sqr()?
        + spectrum.narrow(1, frequencies, frequencies)?.sqr()?)?;
    let filters = Tensor::from_vec(
        filters(BINS, frequencies),
        (BINS, frequencies),
        &Device::Cpu,
    )?;
    let mut values = filters
        .matmul(&power.t()?)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    for v in &mut values {
        *v = v.max(1e-10).log10();
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
