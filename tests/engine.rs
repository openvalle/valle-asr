use anyhow::Result;
use valle_asr::{AsrEngine, AsrModel, Audio, Cancelled, ModelInfo, TranscribeOptions, Transcript};

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
fn cancellation_rejects_pre_cancelled_and_late_success_and_allows_reuse() -> Result<()> {
    struct CancellingBackend(bool);
    impl AsrModel for CancellingBackend {
        fn info(&self) -> ModelInfo {
            Backend("cancel").info()
        }
        fn transcribe(&mut self, audio: &Audio, options: &TranscribeOptions) -> Result<Transcript> {
            if self.0 {
                self.0 = false;
                options.cancellation.cancel();
            }
            Backend("cancel").transcribe(audio, options)
        }
    }
    let mut engine = AsrEngine::new();
    engine.register(CancellingBackend(true))?;
    let audio = Audio::from_mono(vec![0.0; 1600], 16000)?;
    let options = TranscribeOptions::default();
    options.cancellation.cancel();
    assert!(
        engine
            .transcribe("cancel", &audio, &options)
            .unwrap_err()
            .is::<Cancelled>()
    );
    // The rejected request never dispatched the backend: its first actual call
    // still cancels inside inference, and cannot return a successful transcript.
    assert!(
        engine
            .transcribe("cancel", &audio, &TranscribeOptions::default())
            .unwrap_err()
            .is::<Cancelled>()
    );
    assert_eq!(
        engine
            .transcribe("cancel", &audio, &TranscribeOptions::default())?
            .text,
        "cancel"
    );
    Ok(())
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
