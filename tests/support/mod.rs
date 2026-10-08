use anyhow::Result;
use std::{
    fs::File,
    io::{Seek, SeekFrom, Write},
};

/// A standard PCM WAV with a sparse zero-filled payload; no duration-sized PCM
/// vector and no large checked-in binary are required to construct this fixture.
pub fn sparse_wav(file: &mut File, seconds: u32) -> Result<()> {
    let size = seconds * 16000 * 2;
    file.write_all(b"RIFF")?;
    file.write_all(&(size + 36).to_le_bytes())?;
    file.write_all(b"WAVEfmt ")?;
    file.write_all(&16u32.to_le_bytes())?;
    for value in [1u16, 1] {
        file.write_all(&value.to_le_bytes())?;
    }
    for value in [16000u32, 32000] {
        file.write_all(&value.to_le_bytes())?;
    }
    for value in [2u16, 16] {
        file.write_all(&value.to_le_bytes())?;
    }
    file.write_all(b"data")?;
    file.write_all(&size.to_le_bytes())?;
    file.set_len(u64::from(size) + 44)?;
    Ok(())
}

pub fn write_pcm(file: &mut File, offset_samples: u64, samples: &[f32]) -> Result<()> {
    file.seek(SeekFrom::Start(44 + offset_samples * 2))?;
    let mut output = std::io::BufWriter::new(file);
    for sample in samples {
        output.write_all(&((*sample * 32768.0).round() as i16).to_le_bytes())?;
    }
    output.flush()?;
    Ok(())
}
