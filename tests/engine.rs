#[macro_use]
mod support;
use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};
use support::{Result, SparseSource};
use valle_asr::{
    AsrEngine, AsrModel, AudioSource, Error, ModelInfo, Segment, TranscribeOptions, Word,
};

/// Emits one segment per second-long read, covering the whole source.
struct Backend {
    id: &'static str,
    cancel_once: bool,
    largest: Arc<AtomicUsize>,
}

impl Backend {
    fn new(id: &'static str) -> Self {
        Self {
            id,
            cancel_once: false,
            largest: Arc::default(),
        }
    }
}

impl AsrModel for Backend {
    fn info(&self) -> ModelInfo {
        ModelInfo {
            id: self.id.into(),
            family: "test-family".into(),
            word_timestamps: true,
        }
    }

    fn transcribe(
        &mut self,
        audio: &mut dyn AudioSource,
        options: &TranscribeOptions,
        emit: &mut dyn FnMut(Segment) -> valle_asr::Result<()>,
    ) -> valle_asr::Result<()> {
        let step = u64::from(options.chunk_seconds) * 16_000;
        let mut start = 0;
        while start < audio.samples() {
            if std::mem::take(&mut self.cancel_once) {
                options.cancellation.cancel();
            }
            options.cancellation.check()?;
            let count = step.min(audio.samples() - start) as usize;
            let mut samples = vec![0.0; count];
            audio.read(start, &mut samples).map_err(Error::Audio)?;
            self.largest.fetch_max(count, Ordering::SeqCst);
            let (start_ms, end_ms) = (start / 16, (start + count as u64).div_ceil(16));
            emit(Segment {
                text: self.id.into(),
                language: "English".into(),
                start_ms,
                end_ms,
                words: vec![Word {
                    text: self.id.into(),
                    start_ms,
                    end_ms,
                }],
            })?;
            start += count as u64;
        }
        Ok(())
    }
}

#[test]
fn callers_select_between_independently_registered_models() -> Result<()> {
    let mut engine = AsrEngine::new();
    engine.register(Backend::new("first"))?;
    engine.register(Backend::new("second"))?;
    assert!(engine.register(Backend::new("first")).is_err());
    let samples = vec![0.0; 1600];
    for id in ["first", "second"] {
        let transcript = engine.transcribe(id, &samples, &TranscribeOptions::default())?;
        assert_eq!(
            (transcript.text.as_str(), transcript.model.as_str()),
            (id, id)
        );
        assert_eq!(transcript.duration_ms, 100);
    }
    assert_eq!(engine.models().len(), 2);
    assert!(matches!(
        engine.transcribe("missing", &samples, &TranscribeOptions::default()),
        Err(Error::UnknownModel(id)) if id == "missing"
    ));
    assert!(matches!(
        engine.transcribe("first", &[], &TranscribeOptions::default()),
        Err(Error::Invalid(_))
    ));
    Ok(())
}

#[test]
fn cancellation_rejects_pre_cancelled_and_late_success_and_allows_reuse() -> Result<()> {
    let mut engine = AsrEngine::new();
    engine.register(Backend {
        cancel_once: true,
        ..Backend::new("cancel")
    })?;
    let samples = vec![0.0; 1600];
    let options = TranscribeOptions::default();
    options.cancellation.cancel();
    assert!(matches!(
        engine.transcribe("cancel", &samples, &options),
        Err(Error::Cancelled)
    ));
    // The rejected request never dispatched the backend: its first actual call
    // still cancels inside inference, and cannot return a successful transcript.
    assert!(matches!(
        engine.transcribe("cancel", &samples, &TranscribeOptions::default()),
        Err(Error::Cancelled)
    ));
    assert_eq!(
        engine
            .transcribe("cancel", &samples, &TranscribeOptions::default())?
            .text,
        "cancel"
    );
    Ok(())
}

#[test]
fn cancellation_from_the_last_sink_call_is_not_reported_as_success() -> Result<()> {
    let mut engine = AsrEngine::new();
    engine.register(Backend::new("counter"))?;
    let options = TranscribeOptions::default();
    let mut source = SparseSource {
        samples: 16_000,
        inserts: vec![],
    };
    let mut callbacks = 0;
    let error = engine
        .transcribe_source("counter", &mut source, &options, &mut |_| {
            callbacks += 1;
            options.cancellation.cancel();
            Ok(())
        })
        .unwrap_err();
    assert!(matches!(error, Error::Cancelled));
    assert_eq!(callbacks, 1);
    let error = engine
        .transcribe_source(
            "counter",
            &mut source,
            &TranscribeOptions::default(),
            &mut |_| Err("stopped by consumer".into()),
        )
        .unwrap_err();
    assert!(matches!(&error, Error::Sink(e) if e.to_string() == "stopped by consumer"));
    Ok(())
}

#[test]
fn a_two_hour_source_streams_one_chunk_at_a_time() -> Result<()> {
    let largest = Arc::new(AtomicUsize::new(0));
    let mut engine = AsrEngine::new();
    engine.register(Backend {
        largest: largest.clone(),
        ..Backend::new("counter")
    })?;
    let mut source = SparseSource {
        samples: 2 * 60 * 60 * 16_000,
        inserts: vec![(0, vec![0.25])],
    };
    let mut count = 0;
    let mut previous_end = 0;
    let summary = engine.transcribe_source(
        "counter",
        &mut source,
        &TranscribeOptions::default(),
        &mut |segment| {
            ensure!(
                segment.start_ms == previous_end,
                "gap or overlap on the global timeline"
            );
            previous_end = segment.end_ms;
            count += 1;
            Ok(())
        },
    )?;
    ensure!(
        count == 240 && summary.segment_count == 240,
        "wrong two-hour chunk count"
    );
    ensure!(
        summary.duration_ms == 7_200_000 && previous_end == summary.duration_ms,
        "wrong duration"
    );
    ensure!(
        largest.load(Ordering::SeqCst) == 30 * 16000,
        "backend read more than one chunk"
    );
    ensure!(
        summary.text == vec!["counter"; 240].join(" "),
        "summary text lost a segment"
    );
    Ok(())
}

#[test]
fn segments_outside_the_audio_are_rejected() -> Result<()> {
    struct Overrun;
    impl AsrModel for Overrun {
        fn info(&self) -> ModelInfo {
            Backend::new("overrun").info()
        }
        fn transcribe(
            &mut self,
            _: &mut dyn AudioSource,
            _: &TranscribeOptions,
            emit: &mut dyn FnMut(Segment) -> valle_asr::Result<()>,
        ) -> valle_asr::Result<()> {
            emit(Segment {
                text: "late".into(),
                language: "English".into(),
                start_ms: 0,
                end_ms: 101,
                words: vec![],
            })
        }
    }
    let mut engine = AsrEngine::new();
    engine.register(Overrun)?;
    assert!(matches!(
        engine.transcribe("overrun", &[0.0; 1600], &TranscribeOptions::default()),
        Err(Error::Model(_))
    ));
    Ok(())
}
