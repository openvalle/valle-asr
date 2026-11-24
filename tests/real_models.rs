//! CI invokes this ignored test explicitly on every supported OS. Missing models
//! fail the test; it never silently skips inference.
#![cfg(all(feature = "qwen3", feature = "download"))]
use anyhow::{Context, Result, ensure};
mod support;
use std::{
    path::Path,
    time::{Duration, Instant},
};
use support::{SparseSource, fixture};
use valle_asr::{
    AsrModel, AudioSource, CancellationToken, Cancelled, Segment, TranscribeOptions,
    cache::{ModelCache, builtin_model},
    models::qwen3::Qwen3,
};

fn normalized(text: &str) -> Vec<char> {
    text.chars()
        .filter(|c| c.is_alphanumeric())
        .flat_map(char::to_lowercase)
        .collect()
}
fn error_rate(reference: &str, hypothesis: &str) -> f64 {
    let a = normalized(reference);
    let b = normalized(hypothesis);
    let mut row: Vec<_> = (0..=b.len()).collect();
    for (i, ac) in a.iter().enumerate() {
        let mut prev = row[0];
        row[0] = i + 1;
        for (j, bc) in b.iter().enumerate() {
            let old = row[j + 1];
            row[j + 1] = (prev + usize::from(ac != bc)).min(row[j] + 1).min(old + 1);
            prev = old;
        }
    }
    row[b.len()] as f64 / a.len().max(1) as f64
}
fn transcribe(
    backend: &mut Qwen3,
    audio: &mut dyn AudioSource,
    options: &TranscribeOptions,
) -> Result<Vec<Segment>> {
    let mut segments = Vec::new();
    backend.transcribe(audio, options, &mut |segment| {
        segments.push(segment);
        Ok(())
    })?;
    Ok(segments)
}
fn text(segments: &[Segment]) -> String {
    let separator = if segments.iter().any(|s| s.language == "Chinese") {
        ""
    } else {
        " "
    };
    segments
        .iter()
        .map(|s| s.text.as_str())
        .filter(|t| !t.is_empty())
        .collect::<Vec<_>>()
        .join(separator)
}
fn check_timestamps(segments: &[Segment]) -> Result<()> {
    let mut previous = 0;
    for segment in segments {
        ensure!(
            !segment.words.is_empty(),
            "no aligned words: {}",
            segment.text
        );
        for word in &segment.words {
            ensure!(
                word.start_ms >= previous && word.start_ms <= word.end_ms,
                "nonmonotonic word: {word:?}"
            );
            ensure!(
                word.start_ms >= segment.start_ms && word.end_ms <= segment.end_ms,
                "word outside source chunk: {word:?}"
            );
            previous = word.end_ms;
        }
        let first = segment.words.first().unwrap();
        let last = segment.words.last().unwrap();
        ensure!(
            last.end_ms - first.start_ms > (segment.end_ms - segment.start_ms) / 2,
            "timestamp predictions collapsed into a short span"
        );
        ensure!(
            segment
                .words
                .iter()
                .filter(|w| w.end_ms > w.start_ms)
                .count()
                * 2
                >= segment.words.len(),
            "too many zero-length words"
        );
    }
    Ok(())
}
fn save(name: &str, result: &[Segment]) -> Result<()> {
    std::fs::create_dir_all("artifacts")?;
    std::fs::write(
        format!("artifacts/{name}.json"),
        serde_json::to_vec_pretty(result)?,
    )?;
    Ok(())
}

#[test]
#[ignore = "downloads ~3.7 GB of pinned models; CI runs this explicitly after warming its model cache"]
fn qwen3_transcription_and_learned_word_timestamps() -> Result<()> {
    let started = Instant::now();
    let root = std::env::var_os("VALLE_ASR_TEST_CACHE")
        .context("VALLE_ASR_TEST_CACHE must point to verified models")?;
    let cache = ModelCache::new(root);
    let model = cache.ensure(&builtin_model("qwen3-asr-0.6b")?, true)?;
    let aligner = cache.ensure(&builtin_model("qwen3-forced-aligner-0.6b")?, true)?;
    let mut backend = Qwen3::load("qwen3-asr-0.6b", model, Some(aligner))?;
    let fixtures = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
    // Cancel before the first word-timestamp sink callback can run. The file
    // API must stop during its ASR pass, rather than waiting for all recognition
    // and the start of the second, alignment pass. A fresh request below also
    // proves that cancelled lazy loading/inference does not poison the backend.
    let token = CancellationToken::default();
    let options = TranscribeOptions {
        cancellation: token.clone(),
        ..Default::default()
    };
    let mut callbacks = 0;
    let sample5 = fixture(&fixtures.join("sample5.wav"))?;
    let cancelled = std::thread::scope(|scope| {
        scope.spawn(|| {
            std::thread::sleep(Duration::from_millis(50));
            token.cancel();
        });
        backend.transcribe(&mut &sample5[..], &options, &mut |_| {
            callbacks += 1;
            Ok(())
        })
    })
    .unwrap_err();
    ensure!(
        cancelled.is::<Cancelled>() && callbacks == 0,
        "first-pass cancellation was delayed until the result sink: {cancelled}"
    );
    eprintln!("[test] first ASR pass cancelled without waiting for a result callback");
    let mut receipts = Vec::new();
    for (name, lang, threshold) in [("sample1", "en", 0.10), ("sample5", "zh", 0.20)] {
        let reference = std::fs::read_to_string(fixtures.join(format!("{name}.txt")))?;
        let options = TranscribeOptions {
            language: Some(lang.into()),
            chunk_seconds: 20,
            ..Default::default()
        };
        let case_start = Instant::now();
        let samples = fixture(&fixtures.join(format!("{name}.wav")))?;
        let result = transcribe(&mut backend, &mut &samples[..], &options)?;
        save(name, &result)?;
        let result_text = text(&result);
        println!("{name}: {result_text}");
        let cer = error_rate(&reference, &result_text);
        ensure!(
            cer <= threshold,
            "{name} character error rate {cer:.3}: {result_text}"
        );
        check_timestamps(&result)?;
        receipts.push(serde_json::json!({"case":name,"cer":cer,"seconds":case_start.elapsed().as_secs_f64(),"duration_ms":result.last().map_or(0, |s| s.end_ms)}));
    }
    // Two copies in separate chunks verify that the second set of learned local
    // timestamps is offset into the original source timeline.
    let single = fixture(&fixtures.join("sample1.wav"))?;
    let align_token = CancellationToken::default();
    let reference = std::fs::read_to_string(fixtures.join("sample1.txt"))?;
    let cancelled = std::thread::scope(|scope| {
        scope.spawn(|| {
            std::thread::sleep(Duration::from_millis(10));
            align_token.cancel();
        });
        backend.align_with_cancellation(&single, &reference, "English", &align_token)
    })
    .unwrap_err();
    ensure!(
        cancelled.is::<Cancelled>(),
        "alignment cancellation lost its error type: {cancelled}"
    );
    ensure!(
        !backend.align(&single, &reference, "English")?.is_empty(),
        "aligner was not reusable after cancellation"
    );
    eprintln!("[test] alignment cancelled and the resident aligner successfully reused");
    let mut samples = single.clone();
    samples.resize(4 * 16000, 0.0);
    samples.extend_from_slice(&single);
    let result = transcribe(
        &mut backend,
        &mut &samples[..],
        &TranscribeOptions {
            language: Some("en".into()),
            chunk_seconds: 4,
            ..Default::default()
        },
    )?;
    ensure!(result.len() == 2, "expected two source chunks");
    let reference = std::fs::read_to_string(fixtures.join("sample1.txt"))?;
    for segment in &result {
        ensure!(
            error_rate(&reference, &segment.text) <= 0.15,
            "chunk text mismatch: {}",
            segment.text
        );
    }
    check_timestamps(&result)?;
    ensure!(
        result[1].words[0].start_ms >= 4000,
        "second chunk lost its global offset"
    );
    save("chunked-english", &result)?;
    // A long timeline with real speech at both ends exercises chunking, both
    // ASR and alignment passes over a source read twice, and global offsets.
    // Most samples are digital silence; this validates long sources, not two
    // hours of speech CER.
    let total = 7200 * 16000;
    let last_offset = total - single.len() as u64;
    let mut long_source = SparseSource {
        samples: total,
        inserts: vec![(0, single.clone()), (last_offset, single.clone())],
    };
    let long_started = Instant::now();
    let mut speech_segments = 0;
    let mut segment_count = 0;
    let mut previous_word_end = 0;
    let mut previous_end = 0;
    backend.transcribe(
        &mut long_source,
        &TranscribeOptions {
            language: Some("en".into()),
            ..Default::default()
        },
        &mut |segment| {
            ensure!(segment.start_ms == previous_end, "long-file timeline gap");
            previous_end = segment.end_ms;
            segment_count += 1;
            if !segment.text.is_empty() {
                let cer = error_rate(&reference, &segment.text);
                ensure!(cer <= 0.15, "long-file speech CER {cer}: {}", segment.text);
                ensure!(
                    !segment.words.is_empty(),
                    "long-file speech lost word timestamps"
                );
                if speech_segments == 1 {
                    ensure!(
                        segment.words[0].start_ms >= last_offset * 1000 / 16000,
                        "long-file word timestamp lost its two-hour offset"
                    );
                }
                for word in &segment.words {
                    ensure!(
                        word.start_ms >= previous_word_end
                            && word.start_ms <= word.end_ms
                            && word.end_ms <= segment.end_ms,
                        "invalid long-file word timestamp: {word:?}"
                    );
                    previous_word_end = word.end_ms;
                }
                speech_segments += 1;
            } else {
                ensure!(
                    segment.words.is_empty(),
                    "digital silence hallucinated words"
                );
            }
            Ok(())
        },
    )?;
    ensure!(
        speech_segments == 2 && segment_count == 240,
        "long-file speech/chunk count mismatch"
    );
    ensure!(previous_end == 7_200_000, "long-file duration changed");
    receipts.push(serde_json::json!({"case":"two-hour-stream","duration_ms":previous_end,
        "segments":segment_count,"speech_segments":speech_segments,
        "seconds":long_started.elapsed().as_secs_f64(),"mostly_digital_silence":true,"word_offset_passed":true}));
    println!("two-hour source: {segment_count} segments, {speech_segments} with speech");
    std::fs::write(
        "artifacts/receipt.json",
        serde_json::to_vec_pretty(
            &serde_json::json!({"os":std::env::consts::OS,"arch":std::env::consts::ARCH,"elapsed_seconds":started.elapsed().as_secs_f64(),"cases":receipts,"chunk_offset_passed":true}),
        )?,
    )?;
    Ok(())
}
