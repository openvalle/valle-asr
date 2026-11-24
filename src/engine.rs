use crate::{
    AudioSource, ModelInfo, Segment, TranscribeOptions, Transcript, TranscriptSummary,
    audio::duration_ms,
};
use anyhow::{Result, bail, ensure};
use std::collections::BTreeMap;

/// The extension point for model families. No Qwen types appear in this contract.
pub trait AsrModel: Send {
    /// Return the backend identity and capabilities.
    fn info(&self) -> ModelInfo;

    /// Recognize a source of mono 16 kHz samples and pass each segment to
    /// `emit` in source order. Segment and word times are milliseconds on the
    /// source timeline. Backends must honor supported options, check
    /// `options.cancellation` during inference, and return [`crate::Cancelled`]
    /// instead of partial success.
    ///
    /// # Errors
    /// Returns source, inference and sink errors.
    fn transcribe(
        &mut self,
        audio: &mut dyn AudioSource,
        options: &TranscribeOptions,
        emit: &mut dyn FnMut(Segment) -> Result<()>,
    ) -> Result<()>;
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

    /// Recognize mono 16 kHz samples held in memory, keeping every segment.
    ///
    /// # Errors
    /// Rejects invalid options, empty or invalid samples, cancellation and an
    /// unregistered model ID, and propagates backend failures. Cancellation is
    /// [`crate::Cancelled`].
    pub fn transcribe(
        &mut self,
        model: &str,
        samples: &[f32],
        options: &TranscribeOptions,
    ) -> Result<Transcript> {
        let mut source = samples;
        let mut segments = Vec::new();
        let summary = self.transcribe_source(model, &mut source, options, &mut |segment| {
            segments.push(segment);
            Ok(())
        })?;
        Ok(Transcript {
            model: summary.model,
            text: summary.text,
            language: summary.language,
            duration_ms: summary.duration_ms,
            segments,
        })
    }

    /// Recognize a source of any length, passing each segment to `emit` as it
    /// is ready. The caller decides whether segments are displayed, persisted,
    /// or collected in memory. The summary keeps the full text, a few hundred
    /// bytes per 30 seconds of speech.
    ///
    /// # Errors
    /// Rejects an empty source and an unregistered model ID, and propagates
    /// source, inference and sink errors.
    pub fn transcribe_source(
        &mut self,
        model: &str,
        audio: &mut dyn AudioSource,
        options: &TranscribeOptions,
        emit: &mut dyn FnMut(Segment) -> Result<()>,
    ) -> Result<TranscriptSummary> {
        options.validate()?;
        let Some(backend) = self.models.get_mut(model) else {
            bail!("model is not registered: {model}")
        };
        ensure!(audio.samples() > 0, "audio is empty");
        let duration_ms = duration_ms(audio.samples());
        let mut summary = SummaryBuilder::new(model.to_owned(), duration_ms);
        let mut previous_end = 0;
        backend.transcribe(audio, options, &mut |segment| {
            options.cancellation.check()?;
            ensure!(
                previous_end <= segment.start_ms
                    && segment.start_ms <= segment.end_ms
                    && segment.end_ms <= duration_ms,
                "backend segment out of order or outside the audio"
            );
            ensure!(
                segment
                    .words
                    .iter()
                    .all(|word| segment.start_ms <= word.start_ms
                        && word.start_ms <= word.end_ms
                        && word.end_ms <= segment.end_ms),
                "backend word outside its segment"
            );
            previous_end = segment.end_ms;
            summary.observe(&segment);
            emit(segment)?;
            options.cancellation.check()?;
            Ok(())
        })?;
        options.cancellation.check()?;
        Ok(summary.finish())
    }
}

struct SummaryBuilder {
    summary: TranscriptSummary,
    has_speech: bool,
}

impl SummaryBuilder {
    fn new(model: String, duration_ms: u64) -> Self {
        Self {
            summary: TranscriptSummary {
                model,
                text: String::new(),
                language: String::new(),
                duration_ms,
                segment_count: 0,
            },
            has_speech: false,
        }
    }

    // The language of the first segment with speech names the transcript, and
    // so decides whether segment texts are joined with spaces.
    fn observe(&mut self, segment: &Segment) {
        if self.summary.segment_count == 0 || (!self.has_speech && !segment.text.is_empty()) {
            self.summary.language.clone_from(&segment.language);
        }
        if !segment.text.is_empty() {
            if self.has_speech
                && !["Chinese", "Cantonese", "Japanese"].contains(&self.summary.language.as_str())
            {
                self.summary.text.push(' ');
            }
            self.summary.text.push_str(&segment.text);
            self.has_speech = true;
        }
        self.summary.segment_count += 1;
    }

    fn finish(self) -> TranscriptSummary {
        self.summary
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn segment(text: &str, language: &str) -> Segment {
        Segment {
            text: text.into(),
            language: language.into(),
            start_ms: 0,
            end_ms: 1,
            words: vec![],
        }
    }

    #[test]
    fn text_is_joined_by_the_language_of_the_first_speech() {
        let mut chinese = SummaryBuilder::new("m".into(), 1);
        for text in ["", "你好", "", "世界"] {
            chinese.observe(&segment(text, if text.is_empty() { "" } else { "Chinese" }));
        }
        let chinese = chinese.finish();
        assert_eq!(
            (chinese.text.as_str(), chinese.language.as_str()),
            ("你好世界", "Chinese")
        );
        assert_eq!(chinese.segment_count, 4);
        let mut english = SummaryBuilder::new("m".into(), 1);
        for text in ["Hello", "world"] {
            english.observe(&segment(text, "English"));
        }
        assert_eq!(english.finish().text, "Hello world");
    }
}
