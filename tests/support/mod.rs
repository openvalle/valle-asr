#![allow(dead_code)]
use std::path::Path;
use valle_asr::{AudioSource, BoxError, SAMPLE_RATE};

pub type Result<T = (), E = BoxError> = std::result::Result<T, E>;

/// Return the message as an error unless `condition` holds.
macro_rules! ensure {
    ($condition:expr, $($message:tt)+) => {
        let holds: bool = $condition;
        if !holds {
            return Err(format!($($message)+).into());
        }
    };
}

/// Digital silence of any length with a few clips of sound inserted, so a
/// two-hour source needs no duration-sized vector or large checked-in binary.
pub struct SparseSource {
    pub samples: u64,
    pub inserts: Vec<(u64, Vec<f32>)>,
}

impl AudioSource for SparseSource {
    fn samples(&self) -> u64 {
        self.samples
    }

    fn read(&mut self, start: u64, out: &mut [f32]) -> Result {
        ensure!(
            start + out.len() as u64 <= self.samples,
            "read past the end"
        );
        out.fill(0.0);
        let end = start + out.len() as u64;
        for (offset, clip) in &self.inserts {
            let from = start.max(*offset);
            let to = end.min(offset + clip.len() as u64);
            for index in from..to {
                out[(index - start) as usize] = clip[(index - offset) as usize];
            }
        }
        Ok(())
    }
}

/// Read a 16 kHz mono PCM16 fixture.
pub fn fixture(path: &Path) -> Result<Vec<f32>> {
    let mut reader = hound::WavReader::open(path)?;
    let spec = reader.spec();
    ensure!(
        spec.sample_rate == SAMPLE_RATE && spec.channels == 1 && spec.bits_per_sample == 16,
        "fixtures are 16 kHz mono PCM16"
    );
    Ok(reader
        .samples::<i16>()
        .map(|v| v.map(|x| f32::from(x) / 32768.0))
        .collect::<Result<_, _>>()?)
}
