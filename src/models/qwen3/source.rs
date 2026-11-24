use super::{Qwen3, text};
use crate::{
    AudioSource, Segment, TimestampMode, TranscribeOptions, audio::duration_ms, chunks::Chunks,
};
use anyhow::{Context, Result, ensure};

fn audible_bounds(samples: &[f32]) -> Option<std::ops::Range<usize>> {
    let start = samples.iter().position(|sample| *sample != 0.0)?;
    let end = samples.iter().rposition(|sample| *sample != 0.0)? + 1;
    Some(start..end)
}

struct PendingSegment {
    start: u64,
    samples: usize,
    segment: Segment,
}

impl Qwen3 {
    fn chunk_segment(
        &mut self,
        start: u64,
        samples: &[f32],
        options: &TranscribeOptions,
    ) -> Result<Segment> {
        // Exact digital silence needs neither ASR nor alignment and should not
        // hallucinate text. This is deliberately not a general speech detector.
        let (text, language) = if let Some(bounds) = audible_bounds(samples) {
            self.transcribe_chunk(&samples[bounds], options)?
        } else {
            (
                String::new(),
                options
                    .language
                    .as_deref()
                    .map(text::language_name)
                    .transpose()?
                    .unwrap_or_default(),
            )
        };
        Ok(Segment {
            text,
            language,
            start_ms: duration_ms(start),
            end_ms: duration_ms(start + samples.len() as u64),
            words: vec![],
        })
    }

    pub(super) fn transcribe_source(
        &mut self,
        audio: &mut dyn AudioSource,
        options: &TranscribeOptions,
        emit: &mut dyn FnMut(Segment) -> Result<()>,
    ) -> Result<()> {
        options.validate()?;
        if options.timestamps == TimestampMode::Word {
            ensure!(
                self.aligner_dir.is_some(),
                "word timestamps require Qwen3-ForcedAligner"
            );
        }
        let mut chunks = Chunks::new(audio, options.chunk_seconds);
        if options.timestamps == TimestampMode::None {
            while let Some(chunk) = chunks.next(&options.cancellation)? {
                let segment = self.chunk_segment(chunk.start, &chunk.samples, options)?;
                options.cancellation.check()?;
                emit(segment)?;
                options.cancellation.check()?;
            }
            return Ok(());
        }

        // Keep each chunk's interval and text, never its audio or model tensors.
        // The text of a chunk is a few hundred bytes, so two hours of speech
        // holds well under a megabyte here while the audio is read again.
        let mut records = Vec::new();
        while let Some(chunk) = chunks.next(&options.cancellation)? {
            let segment = self.chunk_segment(chunk.start, &chunk.samples, options)?;
            records.push(PendingSegment {
                start: chunk.start,
                samples: chunk.samples.len(),
                segment,
            });
        }
        options.cancellation.check()?;
        self.unload();
        for mut record in records {
            let samples = chunks.read(record.start, record.samples)?;
            let offset_ms = if let Some(bounds) = audible_bounds(&samples) {
                let offset = duration_ms(record.start + bounds.start as u64);
                record.segment.words = self
                    .align_chunk(
                        &samples[bounds],
                        &record.segment.text,
                        &record.segment.language,
                        &options.cancellation,
                    )
                    .with_context(|| format!("align at {} ms", record.segment.start_ms))?;
                offset
            } else {
                record.segment.start_ms
            };
            for word in &mut record.segment.words {
                word.start_ms = (word.start_ms + offset_ms).min(record.segment.end_ms);
                word.end_ms = (word.end_ms + offset_ms).min(record.segment.end_ms);
            }
            options.cancellation.check()?;
            emit(record.segment)?;
            options.cancellation.check()?;
        }
        Ok(())
    }
}
