use crate::{AudioSource, CancellationToken, SAMPLE_RATE};
use anyhow::{Result, ensure};

/// One interval of a source, in samples on its 16 kHz timeline.
pub(crate) struct Chunk {
    pub start: u64,
    pub samples: Vec<f32>,
}

/// Cuts a source into chunks of at most `chunk_seconds`, each ending at a
/// pause near its limit when there is one. No sample is dropped or repeated.
pub(crate) struct Chunks<'a> {
    source: &'a mut dyn AudioSource,
    position: u64,
    max_samples: usize,
}

impl<'a> Chunks<'a> {
    pub fn new(source: &'a mut dyn AudioSource, chunk_seconds: u32) -> Self {
        Self {
            source,
            position: 0,
            max_samples: chunk_seconds as usize * SAMPLE_RATE as usize,
        }
    }

    /// The next chunk; `None` at the end of the source.
    pub fn next(&mut self, cancellation: &CancellationToken) -> Result<Option<Chunk>> {
        cancellation.check()?;
        let total = self.source.samples();
        if self.position >= total {
            return Ok(None);
        }
        let count = (total - self.position).min(self.max_samples as u64) as usize;
        let start = self.position;
        let mut samples = self.read(start, count)?;
        if start + (count as u64) < total {
            samples.truncate(quiet_cut(&samples));
        }
        cancellation.check()?;
        self.position += samples.len() as u64;
        Ok(Some(Chunk { start, samples }))
    }

    /// Read `count` samples from `start` again, for a second pass over a chunk.
    pub fn read(&mut self, start: u64, count: usize) -> Result<Vec<f32>> {
        let mut samples = vec![0.0; count];
        self.source.read(start, &mut samples)?;
        ensure!(
            samples
                .iter()
                .all(|v| v.is_finite() && (-1.0..=1.0).contains(v)),
            "samples must be finite and within [-1, 1]"
        );
        Ok(samples)
    }
}

// Look back at most three seconds (or a quarter of a small chunk) for a 120 ms
// quiet interval. Prefer its midpoint; sustained silence at the hard limit can
// use the entire chunk. Continuous speech falls back to the hard limit.
fn quiet_cut(samples: &[f32]) -> usize {
    let frame = 320;
    let lookback = (3 * SAMPLE_RATE as usize).min(samples.len() / 4);
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_pause_near_the_limit_ends_the_chunk_and_every_sample_is_kept() -> Result<()> {
        let rate = SAMPLE_RATE as usize;
        let pause = 27 * rate..28 * rate;
        let samples: Vec<_> = (0..35 * rate)
            .map(|i| if pause.contains(&i) { 0.0 } else { 0.4 })
            .collect();
        let mut source = &samples[..];
        let mut chunks = Chunks::new(&mut source, 30);
        let token = CancellationToken::default();
        let first = chunks.next(&token)?.unwrap();
        assert!(pause.contains(&first.samples.len()));
        let second = chunks.next(&token)?.unwrap();
        assert_eq!(second.start, first.samples.len() as u64);
        assert_eq!(first.samples.len() + second.samples.len(), samples.len());
        assert!(chunks.next(&token)?.is_none());
        assert_eq!(chunks.read(pause.end as u64, 4)?, [0.4; 4]);
        Ok(())
    }

    #[test]
    fn continuous_sound_is_cut_at_the_limit_and_invalid_samples_are_rejected() -> Result<()> {
        let samples = vec![0.4; 31 * SAMPLE_RATE as usize];
        let mut source = &samples[..];
        let mut chunks = Chunks::new(&mut source, 30);
        let first = chunks.next(&CancellationToken::default())?.unwrap();
        assert_eq!(first.samples.len(), 30 * SAMPLE_RATE as usize);
        let invalid = [0.0, f32::NAN, 1.5];
        let mut source = &invalid[..];
        assert!(
            Chunks::new(&mut source, 30)
                .next(&CancellationToken::default())
                .is_err()
        );
        Ok(())
    }
}
