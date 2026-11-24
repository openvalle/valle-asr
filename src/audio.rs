use crate::BoxError;

/// Sample rate of every input, in hertz. Sources are mono f32 PCM at this rate.
pub const SAMPLE_RATE: u32 = 16_000;

/// Mono 16 kHz f32 samples that can be read from any offset, more than once.
///
/// Word timestamps read each chunk twice, once to recognize it and once to
/// align it, so a source never has to hold a whole file in memory. A decoder
/// can seek, or keep its own decoded copy on disk; a slice is a source as well.
pub trait AudioSource {
    /// Total number of samples.
    fn samples(&self) -> u64;

    /// Fill `out` with the samples starting at `start`. Values must be finite
    /// and within `[-1, 1]`; recognition rejects any other value.
    ///
    /// # Errors
    /// Returns the source's own error when the samples cannot be read; the
    /// request fails with [`crate::Error::Audio`] holding it.
    fn read(&mut self, start: u64, out: &mut [f32]) -> Result<(), BoxError>;
}

impl AudioSource for &[f32] {
    fn samples(&self) -> u64 {
        self.len() as u64
    }

    fn read(&mut self, start: u64, out: &mut [f32]) -> Result<(), BoxError> {
        let samples = usize::try_from(start)
            .ok()
            .and_then(|start| self.get(start..start.checked_add(out.len())?))
            .ok_or("read past the end of the samples")?;
        out.copy_from_slice(samples);
        Ok(())
    }
}

/// Duration of `samples` at 16 kHz, rounded up to the next millisecond.
pub(crate) fn duration_ms(samples: u64) -> u64 {
    (samples * 1000).div_ceil(u64::from(SAMPLE_RATE))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn slices_read_in_place_and_reject_reads_past_their_end() {
        let samples = [0.0, 0.25, 0.5, 0.75];
        let mut source = &samples[..];
        let mut out = [0.0; 2];
        source.read(1, &mut out).unwrap();
        assert_eq!(out, [0.25, 0.5]);
        assert!(source.read(3, &mut out).is_err());
        assert_eq!(duration_ms(16_001), 1001);
    }
}
