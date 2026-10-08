use anyhow::Result;
use valle_asr::{AsrEngine, AsrModel, Audio, ModelInfo, TranscribeOptions, Transcript};

struct Backend(&'static str);
impl AsrModel for Backend {
    fn info(&self) -> ModelInfo {
        ModelInfo {
            id: self.0.into(),
            family: "test-family".into(),
            word_timestamps: false,
        }
    }
    fn transcribe(&mut self, audio: &Audio, _: &TranscribeOptions) -> Result<Transcript> {
        Ok(Transcript {
            model: self.0.into(),
            text: self.0.into(),
            language: "English".into(),
            duration_ms: audio.duration_ms(),
            segments: vec![],
        })
    }
}

#[test]
fn callers_select_between_independently_registered_models() {
    let mut engine = AsrEngine::new();
    engine.register(Backend("first")).unwrap();
    engine.register(Backend("second")).unwrap();
    assert!(engine.register(Backend("first")).is_err());
    let audio = Audio::from_mono(vec![0.0; 1600], 16000).unwrap();
    for id in ["first", "second"] {
        assert_eq!(
            engine
                .transcribe(id, &audio, &TranscribeOptions::default())
                .unwrap()
                .text,
            id
        );
    }
    assert_eq!(engine.models().len(), 2);
    assert!(
        engine
            .transcribe("missing", &audio, &TranscribeOptions::default())
            .is_err()
    );
}
