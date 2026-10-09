use anyhow::{Result, ensure};
mod support;
use std::{
    fs::File,
    io::BufWriter,
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
};
use support::{sparse_wav, write_pcm};
use tempfile::{NamedTempFile, tempdir};
use valle_asr::{
    AsrEngine, AsrModel, Audio, Cancelled, JsonTranscriptWriter, ModelInfo, Segment,
    TranscribeOptions, Transcript, TranscriptSummary, WavChunks, Word,
};

struct CountingBackend(Arc<AtomicUsize>);

#[test]
fn cancellation_from_last_sink_is_not_reported_as_success() -> Result<()> {
    let mut input = NamedTempFile::new()?;
    sparse_wav(input.as_file_mut(), 1)?;
    let mut engine = AsrEngine::new();
    engine.register(CountingBackend(Arc::new(AtomicUsize::new(0))))?;
    let options = TranscribeOptions::default();
    let mut callbacks = 0;
    let error = engine
        .transcribe_file("counter", input.path(), &options, &mut |_| {
            callbacks += 1;
            options.cancellation.cancel();
            Ok(())
        })
        .unwrap_err();
    assert!(error.is::<Cancelled>());
    assert_eq!(callbacks, 1);
    let summary = engine.transcribe_file(
        "counter",
        input.path(),
        &TranscribeOptions::default(),
        &mut |_| Ok(()),
    )?;
    assert_eq!(summary.segment_count, 1);
    Ok(())
}

#[test]
fn pre_cancelled_file_does_not_open_input_or_emit() -> Result<()> {
    let mut engine = AsrEngine::new();
    engine.register(CountingBackend(Arc::new(AtomicUsize::new(0))))?;
    let options = TranscribeOptions::default();
    options.cancellation.cancel();
    let error = engine
        .transcribe_file("counter", "missing.wav", &options, &mut |_| {
            panic!("cancelled output emitted")
        })
        .unwrap_err();
    assert!(error.is::<Cancelled>());
    Ok(())
}
impl AsrModel for CountingBackend {
    fn info(&self) -> ModelInfo {
        ModelInfo {
            id: "counter".into(),
            family: "test".into(),
            word_timestamps: true,
        }
    }
    fn transcribe(&mut self, audio: &Audio, _: &TranscribeOptions) -> Result<Transcript> {
        self.0.fetch_max(audio.samples().len(), Ordering::SeqCst);
        let end_ms = audio.duration_ms();
        let text = "quote\" newline\n反斜杠\\".to_owned();
        Ok(Transcript {
            model: "counter".into(),
            language: "English".into(),
            text: text.clone(),
            duration_ms: end_ms,
            segments: vec![Segment {
                text: text.clone(),
                language: "English".into(),
                start_ms: 0,
                end_ms,
                words: vec![Word {
                    text,
                    start_ms: 0,
                    end_ms,
                }],
            }],
        })
    }
}

#[test]
fn two_hour_file_streams_through_engine_and_json_without_retaining_pcm() -> Result<()> {
    let directory = tempdir()?;
    let input_path = directory.path().join("two-hours.wav");
    let mut input = File::create(&input_path)?;
    sparse_wav(&mut input, 2 * 60 * 60)?;
    write_pcm(&mut input, 0, &[0.25])?;
    drop(input);
    let largest = Arc::new(AtomicUsize::new(0));
    let mut engine = AsrEngine::new();
    engine.register(CountingBackend(largest.clone()))?;
    let output = NamedTempFile::new_in(directory.path())?;
    let mut writer = JsonTranscriptWriter::new(BufWriter::new(output.as_file()))?;
    let mut count = 0;
    let mut previous_end = 0;
    let summary = engine.transcribe_file(
        "counter",
        &input_path,
        &TranscribeOptions::default(),
        &mut |segment| {
            ensure!(
                segment.start_ms == previous_end,
                "gap or overlap on the global timeline"
            );
            ensure!(
                segment.words[0].start_ms == segment.start_ms
                    && segment.words[0].end_ms == segment.end_ms,
                "word offsets lost"
            );
            previous_end = segment.end_ms;
            count += 1;
            writer.write_segment(segment)
        },
    )?;
    writer.finish(&summary)?;
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
        "backend received more than one chunk"
    );
    // The generated JSON is small here; production never reads it back in RAM.
    let parsed: Transcript = serde_json::from_reader(File::open(output.path())?)?;
    ensure!(
        parsed.segments.len() == 240 && parsed.duration_ms == 7_200_000,
        "invalid streamed JSON"
    );
    let expected = std::iter::repeat_n("quote\" newline\n反斜杠\\", 240)
        .collect::<Vec<_>>()
        .join(" ");
    ensure!(
        parsed.text == expected,
        "JSON string escaping or separators changed"
    );
    println!(
        "two-hour WAV: 230400044 input bytes, 240 segments, largest normalized chunk {} bytes",
        largest.load(Ordering::SeqCst) * 4
    );
    Ok(())
}

#[test]
fn resampling_is_continuous_across_chunks_and_matches_whole_clip() -> Result<()> {
    for rate in [22050, 44100, 48000] {
        let file = NamedTempFile::new()?;
        let spec = hound::WavSpec {
            channels: 2,
            sample_rate: rate,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };
        let mut writer = hound::WavWriter::new(file.reopen()?, spec)?;
        for index in 0..(rate * 2 + 373) {
            let sample = (12000.0 * (index as f64 * 0.15).sin()) as i16;
            writer.write_sample(sample)?;
            writer.write_sample(sample / 2)?;
        }
        writer.finalize()?;
        let expected = Audio::from_wav(file.path())?;
        let mut reader = WavChunks::open(file.path(), 1)?;
        let mut actual = vec![];
        let mut previous_ms = 0;
        while let Some(chunk) = reader.next_chunk()? {
            ensure!(
                chunk.start_sample == actual.len() as u64 && chunk.start_ms() == previous_ms,
                "resampling timeline discontinuity"
            );
            previous_ms = chunk.end_ms();
            actual.extend_from_slice(chunk.audio.samples());
        }
        ensure!(
            actual.len() == expected.samples().len(),
            "resampling length changed"
        );
        let error = actual
            .iter()
            .zip(expected.samples())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        ensure!(error < 2e-6, "{rate} Hz resampling seam/error: {error}");
        reader.rewind();
        ensure!(
            reader.next_chunk()?.unwrap().start_sample == 0,
            "rewind failed"
        );
    }
    Ok(())
}

#[test]
fn quiet_boundary_preserves_every_sample_and_is_reproducible() -> Result<()> {
    let file = NamedTempFile::new()?;
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: 16000,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut writer = hound::WavWriter::new(file.reopen()?, spec)?;
    let pause = (27 * 16000)..(28 * 16000);
    for index in 0..(35 * 16000) {
        writer.write_sample(if pause.contains(&index) { 0i16 } else { 12000 })?;
    }
    writer.finalize()?;
    let mut reader = WavChunks::open(file.path(), 30)?;
    let first = reader.next_chunk()?.unwrap();
    ensure!(
        pause.contains(&first.audio.samples().len()),
        "boundary did not choose the pause"
    );
    let first_len = first.audio.samples().len();
    let second = reader.next_chunk()?.unwrap();
    ensure!(
        second.start_sample == first_len as u64,
        "boundary dropped or duplicated samples"
    );
    ensure!(
        first_len + second.audio.samples().len() == 35 * 16000,
        "wrong total samples"
    );
    reader.rewind();
    ensure!(
        reader.next_chunk()?.unwrap().audio.samples().len() == first_len,
        "second-pass boundary changed"
    );
    Ok(())
}

#[test]
fn padded_pcm_and_float_wav_decode_with_correct_container_seeks() -> Result<()> {
    let file = NamedTempFile::new()?;
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: 16000,
        bits_per_sample: 24,
        sample_format: hound::SampleFormat::Int,
    };
    let mut writer = hound::WavWriter::new_with_spec_ex(
        file.reopen()?,
        hound::WavSpecEx {
            spec,
            bytes_per_sample: 4,
        },
    )?;
    for index in 0..33000 {
        writer.write_sample(index * 71)?;
    }
    writer.finalize()?;
    let expected = Audio::from_wav(file.path())?;
    let mut chunks = WavChunks::open(file.path(), 1)?;
    let mut samples = vec![];
    while let Some(chunk) = chunks.next_chunk()? {
        samples.extend_from_slice(chunk.audio.samples());
    }
    ensure!(
        samples == expected.samples(),
        "padded PCM seek used valid bits instead of container width"
    );
    let float = NamedTempFile::new()?;
    let mut writer = hound::WavWriter::new(
        float.reopen()?,
        hound::WavSpec {
            bits_per_sample: 32,
            sample_format: hound::SampleFormat::Float,
            ..spec
        },
    )?;
    for _ in 0..17000 {
        writer.write_sample(0.25f32)?;
    }
    writer.finalize()?;
    let mut chunks = WavChunks::open(float.path(), 1)?;
    while let Some(chunk) = chunks.next_chunk()? {
        ensure!(
            chunk.audio.samples().iter().all(|x| *x == 0.25),
            "float decode changed"
        );
    }
    Ok(())
}

#[test]
fn truncated_input_and_sink_failure_are_reported() -> Result<()> {
    let mut input = NamedTempFile::new()?;
    sparse_wav(input.as_file_mut(), 2)?;
    let mut engine = AsrEngine::new();
    engine.register(CountingBackend(Arc::new(AtomicUsize::new(0))))?;
    let error = engine
        .transcribe_file(
            "counter",
            input.path(),
            &TranscribeOptions::default(),
            &mut |_| anyhow::bail!("cancelled by consumer"),
        )
        .unwrap_err();
    ensure!(
        error.to_string().contains("cancelled"),
        "sink error swallowed"
    );
    input.as_file_mut().set_len(100)?;
    ensure!(
        WavChunks::open(input.path(), 1).is_err(),
        "truncated WAV accepted"
    );
    Ok(())
}

#[test]
fn streaming_json_keeps_chinese_text_contiguous() -> Result<()> {
    let mut writer = JsonTranscriptWriter::new(vec![])?;
    for text in ["你好", "世界"] {
        writer.write_segment(Segment {
            text: text.into(),
            language: "Chinese".into(),
            start_ms: 0,
            end_ms: 1,
            words: vec![],
        })?;
    }
    let output = writer.finish(&TranscriptSummary {
        model: "counter".into(),
        language: "Chinese".into(),
        duration_ms: 1,
        segment_count: 2,
    })?;
    let parsed: Transcript = serde_json::from_slice(&output)?;
    ensure!(
        parsed.text == "你好世界",
        "unexpected separator in Chinese text"
    );
    Ok(())
}
