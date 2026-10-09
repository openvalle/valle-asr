use crate::{
    Audio, ModelInfo, Segment, TranscribeOptions, Transcript, TranscriptSummary, WavChunks,
};
use anyhow::{Result, bail, ensure};
use std::{collections::BTreeMap, path::Path};

/// The extension point for model families. No Qwen types appear in this contract.
pub trait AsrModel: Send {
    /// Return the backend identity and capabilities.
    fn info(&self) -> ModelInfo;
    /// Recognize in-memory audio. Segment and word times are relative to this input.
    /// Backends must honor supported options and return errors for failed inference.
    fn transcribe(&mut self, audio: &Audio, options: &TranscribeOptions) -> Result<Transcript>;

    /// Process one complete WAV with bounded audio buffers, delivering results
    /// to a fallible sink. Backends can override this to optimize model residency.
    fn transcribe_file(
        &mut self,
        path: &Path,
        options: &TranscribeOptions,
        emit: &mut dyn FnMut(Segment) -> Result<()>,
    ) -> Result<TranscriptSummary> {
        let mut chunks = WavChunks::open(path, options.chunk_seconds)?;
        let mut summary = SummaryBuilder::new(self.info().id, chunks.duration_ms());
        while let Some(chunk) = chunks.next_chunk()? {
            let result = self.transcribe(&chunk.audio, options)?;
            let segments = if result.segments.is_empty() {
                vec![Segment {
                    text: result.text,
                    language: result.language,
                    start_ms: 0,
                    end_ms: chunk.audio.duration_ms(),
                    words: vec![],
                }]
            } else {
                result.segments
            };
            for mut segment in segments {
                ensure!(
                    segment.start_ms <= segment.end_ms
                        && segment.end_ms <= chunk.audio.duration_ms(),
                    "backend segment outside its source chunk"
                );
                segment.start_ms += chunk.start_ms();
                segment.end_ms = (segment.end_ms + chunk.start_ms()).min(chunk.end_ms());
                for word in &mut segment.words {
                    word.start_ms = (word.start_ms + chunk.start_ms()).min(segment.end_ms);
                    word.end_ms = (word.end_ms + chunk.start_ms()).min(segment.end_ms);
                }
                summary.observe(&segment);
                emit(segment)?;
            }
        }
        Ok(summary.finish())
    }
}

#[derive(Default)]
/// A registry that dispatches requests to independently registered models.
pub struct AsrEngine {
    models: BTreeMap<String, Box<dyn AsrModel>>,
}

impl AsrEngine {
    /// Create an empty model registry.
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a model under its reported ID.
    ///
    /// # Errors
    /// Returns an error if the ID is empty or already registered.
    pub fn register(&mut self, model: impl AsrModel + 'static) -> Result<()> {
        let id = model.info().id;
        ensure!(!id.is_empty(), "model ID is empty");
        ensure!(
            !self.models.contains_key(&id),
            "model already registered: {id}"
        );
        self.models.insert(id, Box::new(model));
        Ok(())
    }

    /// Return model capabilities in ascending model-ID order.
    pub fn models(&self) -> Vec<ModelInfo> {
        self.models.values().map(|m| m.info()).collect()
    }

    /// Recognize in-memory audio with the selected backend.
    ///
    /// # Errors
    /// Rejects an unregistered model ID and propagates backend failures.
    pub fn transcribe(
        &mut self,
        model: &str,
        audio: &Audio,
        options: &TranscribeOptions,
    ) -> Result<Transcript> {
        let Some(backend) = self.models.get_mut(model) else {
            bail!("model is not registered: {model}")
        };
        backend.transcribe(audio, options)
    }

    /// Stream a complete file through the selected backend. The caller decides
    /// whether segments are displayed, persisted, or collected in memory.
    ///
    /// # Errors
    /// Rejects an unregistered model ID and propagates decoding, inference and sink errors.
    pub fn transcribe_file(
        &mut self,
        model: &str,
        path: impl AsRef<Path>,
        options: &TranscribeOptions,
        emit: &mut dyn FnMut(Segment) -> Result<()>,
    ) -> Result<TranscriptSummary> {
        let Some(backend) = self.models.get_mut(model) else {
            bail!("model is not registered: {model}")
        };
        backend.transcribe_file(path.as_ref(), options, emit)
    }
}

pub(crate) struct SummaryBuilder {
    summary: TranscriptSummary,
    has_speech: bool,
}

impl SummaryBuilder {
    pub(crate) fn new(model: String, duration_ms: u64) -> Self {
        Self {
            summary: TranscriptSummary {
                model,
                language: String::new(),
                duration_ms,
                segment_count: 0,
            },
            has_speech: false,
        }
    }

    pub(crate) fn observe(&mut self, segment: &Segment) {
        if self.summary.segment_count == 0 || (!self.has_speech && !segment.text.is_empty()) {
            self.summary.language.clone_from(&segment.language);
        }
        self.has_speech |= !segment.text.is_empty();
        self.summary.segment_count += 1;
    }

    pub(crate) fn finish(self) -> TranscriptSummary {
        self.summary
    }
}
