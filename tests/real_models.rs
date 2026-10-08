//! CI invokes this ignored test explicitly on every supported OS. Missing models
//! fail the test; it never silently skips inference.
#![cfg(all(feature = "qwen3", feature = "download"))]
use anyhow::{Context, Result, ensure};
use std::{path::Path, time::Instant};
use valle_asr::{
    AsrModel, Audio, TranscribeOptions, Transcript,
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
fn check_timestamps(result: &Transcript) -> Result<()> {
    let mut previous = 0;
    for segment in &result.segments {
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
fn save(name: &str, result: &Transcript) -> Result<()> {
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
    let mut receipts = Vec::new();
    for (name, lang, threshold) in [("sample1", "en", 0.10), ("sample5", "zh", 0.20)] {
        let audio = Audio::from_wav(fixtures.join(format!("{name}.wav")))?;
        let reference = std::fs::read_to_string(fixtures.join(format!("{name}.txt")))?;
        let options = TranscribeOptions {
            language: Some(lang.into()),
            chunk_seconds: 20,
            ..Default::default()
        };
        let case_start = Instant::now();
        let result = backend.transcribe(&audio, &options)?;
        println!("{name}: {}", result.text);
        let cer = error_rate(&reference, &result.text);
        ensure!(
            cer <= threshold,
            "{name} character error rate {cer:.3}: {}",
            result.text
        );
        check_timestamps(&result)?;
        save(name, &result)?;
        receipts.push(serde_json::json!({"case":name,"cer":cer,"seconds":case_start.elapsed().as_secs_f64(),"duration_ms":audio.duration_ms()}));
    }
    // Two copies in separate chunks verify that the second set of learned local
    // timestamps is offset into the original source timeline.
    let single = Audio::from_wav(fixtures.join("sample1.wav"))?;
    let mut samples = single.samples().to_vec();
    samples.resize(4 * 16000, 0.0);
    samples.extend_from_slice(single.samples());
    let audio = Audio::from_mono(samples, 16000)?;
    let result = backend.transcribe(
        &audio,
        &TranscribeOptions {
            language: Some("en".into()),
            chunk_seconds: 4,
            ..Default::default()
        },
    )?;
    ensure!(result.segments.len() == 2, "expected two source chunks");
    let reference = std::fs::read_to_string(fixtures.join("sample1.txt"))?;
    for segment in &result.segments {
        ensure!(
            error_rate(&reference, &segment.text) <= 0.15,
            "chunk text mismatch: {}",
            segment.text
        );
    }
    check_timestamps(&result)?;
    ensure!(
        result.segments[1].words[0].start_ms >= 4000,
        "second chunk lost its global offset"
    );
    save("chunked-english", &result)?;
    std::fs::write(
        "artifacts/receipt.json",
        serde_json::to_vec_pretty(
            &serde_json::json!({"os":std::env::consts::OS,"arch":std::env::consts::ARCH,"elapsed_seconds":started.elapsed().as_secs_f64(),"cases":receipts,"chunk_offset_passed":true}),
        )?,
    )?;
    Ok(())
}
