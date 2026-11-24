use super::{Qwen3, text};
use crate::{
    AudioChunk, Segment, TimestampMode, TranscribeOptions, TranscriptSummary, WavChunks,
    engine::SummaryBuilder,
};
use anyhow::{Context, Result, ensure};
use std::path::Path;

fn audible_bounds(samples: &[f32]) -> Option<std::ops::Range<usize>> {
    let start = samples.iter().position(|sample| *sample != 0.0)?;
    let end = samples.iter().rposition(|sample| *sample != 0.0)? + 1;
    Some(start..end)
}

struct PendingSegment {
    start_sample: u64,
    samples: usize,
    segment: Segment,
}

impl Qwen3 {
    fn file_segment(&mut self, chunk: &AudioChunk, options: &TranscribeOptions) -> Result<Segment> {
        // Exact digital silence needs neither ASR nor alignment and should not
        // hallucinate text. This is deliberately not a general speech detector.
        let (text, language) = if let Some(bounds) = audible_bounds(chunk.audio.samples()) {
            self.transcribe_chunk(&chunk.audio.samples()[bounds], options)?
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
            start_ms: chunk.start_ms(),
            end_ms: chunk.end_ms(),
            words: vec![],
        })
    }

    pub(super) fn transcribe_wav(
        &mut self,
        path: &Path,
        options: &TranscribeOptions,
        emit: &mut dyn FnMut(Segment) -> Result<()>,
    ) -> Result<TranscriptSummary> {
        options.validate()?;
        if options.timestamps == TimestampMode::Word {
            ensure!(
                self.aligner_dir.is_some(),
                "word timestamps require Qwen3-ForcedAligner"
            );
        }
        let mut chunks = WavChunks::open(path, options.chunk_seconds)?;
        let mut summary = SummaryBuilder::new(self.id.clone(), chunks.duration_ms());
        if options.timestamps == TimestampMode::None {
            while let Some(chunk) = chunks.next_chunk_with_cancellation(&options.cancellation)? {
                let segment = self.file_segment(&chunk, options)?;
                summary.observe(&segment);
                options.cancellation.check()?;
                emit(segment)?;
                options.cancellation.check()?;
            }
            return Ok(summary.finish());
        }

        // Keep each chunk's interval and text, never its audio or model tensors.
        // The text of a chunk is a few hundred bytes, so two hours of speech
        // holds well under a megabyte here while the audio is read again.
        let mut records = Vec::new();
        while let Some(chunk) = chunks.next_chunk_with_cancellation(&options.cancellation)? {
            let segment = self.file_segment(&chunk, options)?;
            summary.observe(&segment);
            records.push(PendingSegment {
                start_sample: chunk.start_sample,
                samples: chunk.audio.samples().len(),
                segment,
            });
        }
        options.cancellation.check()?;
        self.unload();
        chunks.rewind();
        for mut record in records {
            let chunk = chunks
                .next_chunk_with_cancellation(&options.cancellation)?
                .context("WAV changed between ASR and alignment passes")?;
            ensure!(
                chunk.start_sample == record.start_sample
                    && chunk.audio.samples().len() == record.samples,
                "WAV chunk boundaries changed between passes"
            );
            let offset_ms = if let Some(bounds) = audible_bounds(chunk.audio.samples()) {
                let offset = ((chunk.start_sample + bounds.start as u64) * 1000)
                    .div_ceil(u64::from(crate::Audio::SAMPLE_RATE));
                record.segment.words = self.align_chunk(
                    &chunk.audio.samples()[bounds],
                    &record.segment.text,
                    &record.segment.language,
                    &options.cancellation,
                )?;
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
        ensure!(
            chunks
                .next_chunk_with_cancellation(&options.cancellation)?
                .is_none(),
            "WAV grew between ASR and alignment passes"
        );
        options.cancellation.check()?;
        Ok(summary.finish())
    }
}
