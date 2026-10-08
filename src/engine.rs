use crate::{Audio, ModelInfo, TranscribeOptions, Transcript};
use anyhow::{Result, bail, ensure};
use std::collections::BTreeMap;

/// The extension point for model families. No Qwen types appear in this contract.
pub trait AsrModel: Send {
    fn info(&self) -> ModelInfo;
    fn transcribe(&mut self, audio: &Audio, options: &TranscribeOptions) -> Result<Transcript>;
}

#[derive(Default)]
pub struct AsrEngine {
    models: BTreeMap<String, Box<dyn AsrModel>>,
}

impl AsrEngine {
    pub fn new() -> Self {
        Self::default()
    }

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

    pub fn models(&self) -> Vec<ModelInfo> {
        self.models.values().map(|m| m.info()).collect()
    }

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
}
