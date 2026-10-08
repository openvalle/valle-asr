use super::{Qwen3, text};
use crate::{
    AudioChunk, Segment, TimestampMode, TranscribeOptions, TranscriptSummary, WavChunks,
    engine::SummaryBuilder,
};
use anyhow::{Context, Result, ensure};
use serde::{Deserialize, Serialize};
use std::{
    io::{BufReader, BufWriter, Seek, SeekFrom, Write},
    path::Path,
};
use tempfile::NamedTempFile;

fn audible_bounds(samples: &[f32]) -> Option<std::ops::Range<usize>> {
    let start = samples.iter().position(|sample| *sample != 0.0)?;
    let end = samples.iter().rposition(|sample| *sample != 0.0)? + 1;
    Some(start..end)
}

#[derive(Serialize, Deserialize)]
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
        ensure!(
            options.max_new_tokens > 0,
            "max_new_tokens must be positive"
        );
        if options.timestamps == TimestampMode::Word {
            ensure!(
                self.aligner_dir.is_some(),
                "word timestamps require Qwen3-ForcedAligner"
            );
        }
        let mut chunks = WavChunks::open(path, options.chunk_seconds)?;
        let mut summary = SummaryBuilder::new(self.id.clone(), chunks.duration_ms());
        if options.timestamps == TimestampMode::None {
            while let Some(chunk) = chunks.next_chunk()? {
                let segment = self.file_segment(&chunk, options)?;
                summary.observe(&segment);
                emit(segment)?;
            }
            return Ok(summary.finish());
        }

        // Store text and interval metadata, never audio or model tensors. This
        // keeps both model residency and transcript RAM independent of duration.
        let mut spool = NamedTempFile::new()?;
        {
            let mut records = BufWriter::new(spool.as_file_mut());
            while let Some(chunk) = chunks.next_chunk()? {
                let segment = self.file_segment(&chunk, options)?;
                summary.observe(&segment);
                let record = PendingSegment {
                    start_sample: chunk.start_sample,
                    samples: chunk.audio.samples().len(),
                    segment,
                };
                serde_json::to_writer(&mut records, &record)?;
                records.write_all(b"\n")?;
            }
            records.flush()?;
        }
        self.unload();
        chunks.rewind();
        spool.as_file_mut().seek(SeekFrom::Start(0))?;
        let records = serde_json::Deserializer::from_reader(BufReader::new(spool.as_file_mut()))
            .into_iter::<PendingSegment>();
        for record in records {
            let mut record = record?;
            let chunk = chunks
                .next_chunk()?
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
                )?;
                offset
            } else {
                record.segment.start_ms
            };
            for word in &mut record.segment.words {
                word.start_ms = (word.start_ms + offset_ms).min(record.segment.end_ms);
                word.end_ms = (word.end_ms + offset_ms).min(record.segment.end_ms);
            }
            emit(record.segment)?;
        }
        ensure!(
            chunks.next_chunk()?.is_none(),
            "WAV grew between ASR and alignment passes"
        );
        Ok(summary.finish())
    }
}
