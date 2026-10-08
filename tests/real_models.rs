//! CI invokes this ignored test explicitly on every supported OS. Missing models
//! fail the test; it never silently skips inference.
#![cfg(all(feature = "qwen3", feature = "download"))]
use anyhow::{Context, Result, ensure};
mod support;
use std::{path::Path, time::Instant};
use valle_asr::{
    AsrModel, Audio, JsonTranscriptWriter, TranscribeOptions, Transcript,
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
        let reference = std::fs::read_to_string(fixtures.join(format!("{name}.txt")))?;
        let options = TranscribeOptions {
            language: Some(lang.into()),
            chunk_seconds: 20,
            ..Default::default()
        };
        let case_start = Instant::now();
        std::fs::create_dir_all("artifacts")?;
        let output_path = format!("artifacts/{name}.json");
        let mut writer = JsonTranscriptWriter::new(std::io::BufWriter::new(
            std::fs::File::create(&output_path)?,
        ))?;
        let summary = backend.transcribe_file(
            &fixtures.join(format!("{name}.wav")),
            &options,
            &mut |segment| writer.write_segment(segment),
        )?;
        writer.finish(&summary)?;
        let result: Transcript = serde_json::from_reader(std::fs::File::open(&output_path)?)?;
        ensure!(
            result.duration_ms == summary.duration_ms
                && result.segments.len() as u64 == summary.segment_count,
            "streamed JSON summary mismatch"
        );
        println!("{name}: {}", result.text);
        let cer = error_rate(&reference, &result.text);
        ensure!(
            cer <= threshold,
            "{name} character error rate {cer:.3}: {}",
            result.text
        );
        check_timestamps(&result)?;
        receipts.push(serde_json::json!({"case":name,"cer":cer,"seconds":case_start.elapsed().as_secs_f64(),"duration_ms":result.duration_ms}));
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
    // A long timeline with real speech at both ends exercises the file decoder,
    // disk-spooled ASR/alignment passes and JSON sink. Most samples are digital
    // silence; this validates long-file handling, not two hours of speech CER.
    let mut long_file = tempfile::NamedTempFile::new()?;
    support::sparse_wav(long_file.as_file_mut(), 2 * 60 * 60)?;
    let last_offset = 7200 * 16000 - single.samples().len() as u64;
    support::write_pcm(long_file.as_file_mut(), 0, single.samples())?;
    support::write_pcm(long_file.as_file_mut(), last_offset, single.samples())?;
    let long_started = Instant::now();
    let output = std::fs::File::create("artifacts/two-hour-stream.json")?;
    let mut writer = JsonTranscriptWriter::new(std::io::BufWriter::new(output))?;
    let mut speech_segments = 0;
    let mut previous_word_end = 0;
    let long_summary = backend.transcribe_file(
        long_file.path(),
        &TranscribeOptions {
            language: Some("en".into()),
            ..Default::default()
        },
        &mut |segment| {
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
            writer.write_segment(segment)
        },
    )?;
    writer.finish(&long_summary)?;
    ensure!(
        speech_segments == 2 && long_summary.segment_count == 240,
        "long-file speech/chunk count mismatch"
    );
    ensure!(
        long_summary.duration_ms == 7_200_000,
        "long-file duration changed"
    );
    receipts.push(serde_json::json!({"case":"two-hour-stream","duration_ms":long_summary.duration_ms,
        "segments":long_summary.segment_count,"speech_segments":speech_segments,
        "seconds":long_started.elapsed().as_secs_f64(),"mostly_digital_silence":true,"word_offset_passed":true}));
    println!("two-hour streamed file: {long_summary:?}");
    std::fs::write(
        "artifacts/receipt.json",
        serde_json::to_vec_pretty(
            &serde_json::json!({"os":std::env::consts::OS,"arch":std::env::consts::ARCH,"elapsed_seconds":started.elapsed().as_secs_f64(),"cases":receipts,"chunk_offset_passed":true}),
        )?,
    )?;
    Ok(())
}
