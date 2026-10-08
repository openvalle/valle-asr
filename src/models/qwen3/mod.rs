//! Qwen3-ASR and its separate learned timestamp classifier, running on CPU.
mod config;
mod dsp;
mod network;
mod text;

use crate::{
    AsrModel, Audio, ModelInfo, Segment, TimestampMode, TranscribeOptions, Transcript, Word,
};
use anyhow::{Context, Result, bail, ensure};
use candle_core::{IndexOp, Tensor};
use network::Network;
use std::path::{Path, PathBuf};
use tokenizers::Tokenizer;

#[derive(Clone, Copy, PartialEq, Eq)]
enum Kind {
    Asr,
    Aligner,
}

/// Lazy model with a bounded resident cache. ASR runs for all chunks first,
/// then its weights are released before the aligner loads. The final session
/// stays cached for repeated text-only transcription or standalone alignment.
pub struct Qwen3 {
    id: String,
    model_dir: PathBuf,
    aligner_dir: Option<PathBuf>,
    asr_tokenizer: Tokenizer,
    aligner_tokenizer: Option<Tokenizer>,
    resident: Option<(Kind, Network)>,
}

impl Qwen3 {
    pub fn load(
        id: impl Into<String>,
        model_dir: impl AsRef<Path>,
        aligner_dir: Option<PathBuf>,
    ) -> Result<Self> {
        let model_dir = model_dir.as_ref().to_path_buf();
        let config: config::Config =
            serde_json::from_slice(&std::fs::read(model_dir.join("config.json"))?)?;
        config.validate()?;
        ensure!(
            config.thinker_config.classify_num.is_none(),
            "ASR path points to a forced aligner"
        );
        let aligner_tokenizer = aligner_dir
            .as_ref()
            .map(|path| text::tokenizer(path))
            .transpose()?;
        Ok(Self {
            id: id.into(),
            asr_tokenizer: text::tokenizer(&model_dir)?,
            model_dir,
            aligner_dir,
            aligner_tokenizer,
            resident: None,
        })
    }

    /// Explicitly release the resident neural network; tokenizers remain cached.
    pub fn unload(&mut self) {
        self.resident = None;
    }

    fn session(&mut self, kind: Kind) -> Result<&Network> {
        if self
            .resident
            .as_ref()
            .is_none_or(|(loaded, _)| *loaded != kind)
        {
            self.resident = None; // release old weights before allocating the new model
            let path = match kind {
                Kind::Asr => &self.model_dir,
                Kind::Aligner => self
                    .aligner_dir
                    .as_ref()
                    .context("word timestamps require a Qwen3-ForcedAligner model")?,
            };
            let network =
                Network::load(path).with_context(|| format!("load {}", path.display()))?;
            ensure!(
                (kind == Kind::Aligner) == network.config.thinker_config.classify_num.is_some(),
                "incorrect model kind"
            );
            self.resident = Some((kind, network));
        }
        Ok(&self.resident.as_ref().context("missing model session")?.1)
    }

    fn transcribe_chunk(
        &mut self,
        samples: &[f32],
        options: &TranscribeOptions,
    ) -> Result<(String, String)> {
        let tok = self.asr_tokenizer.clone();
        let network = self.session(Kind::Asr)?;
        let audio = network.audio(samples)?;
        let c = &network.config.thinker_config;
        let mut ids = text::encode(
            &tok,
            &format!(
                "<|im_start|>system\n{}<|im_end|>\n<|im_start|>user\n",
                options.context
            ),
        )?;
        ids.push(c.audio_start_token_id);
        let audio_start = ids.len();
        ids.extend(std::iter::repeat_n(c.audio_token_id, audio.dim(0)?));
        ids.push(c.audio_end_token_id);
        ids.extend(text::encode(&tok, "<|im_end|>\n<|im_start|>assistant\n")?);
        if let Some(language) = &options.language {
            ids.extend(text::encode(
                &tok,
                &format!("language {}<asr_text>", text::language_name(language)?),
            )?);
        }
        let x = splice(network, &ids, audio_start, &audio)?;
        let mut cache = network.cache();
        let mut logits = network.decode(&x, 0, &mut cache, true)?;
        let mut output = Vec::new();
        for step in 0..options.max_new_tokens {
            let next = logits.flatten_all()?.argmax(0)?.to_scalar::<u32>()?;
            if [151643, 151645].contains(&next) {
                let decoded = tok
                    .decode(&output, false)
                    .map_err(|e| anyhow::anyhow!("decode: {e}"))?;
                return text::parse_output(&decoded, options.language.as_deref());
            }
            output.push(next);
            if step + 1 < options.max_new_tokens {
                logits =
                    network.decode(&network.embed(&[next])?, ids.len() + step, &mut cache, true)?;
            }
        }
        bail!(
            "Qwen3 decode reached max_new_tokens without an end token; increase the limit or shorten chunks"
        )
    }

    /// Align a known transcript to a single clip of at most 30 seconds.
    pub fn align(&mut self, audio: &Audio, transcript: &str, language: &str) -> Result<Vec<Word>> {
        ensure!(
            audio.samples().len() <= 30 * Audio::SAMPLE_RATE as usize,
            "alignment clips must be at most 30 seconds"
        );
        self.align_chunk(audio.samples(), transcript, language)
    }

    fn align_chunk(
        &mut self,
        samples: &[f32],
        transcript: &str,
        language: &str,
    ) -> Result<Vec<Word>> {
        if transcript.trim().is_empty() {
            return Ok(Vec::new());
        }
        let language = text::language_name(language)?;
        // Japanese and Korean require distinct morphological tokenizers. Do not
        // label character/whitespace heuristics as official word segmentation.
        ensure!(
            !["Japanese", "Korean"].contains(&language.as_str()),
            "Japanese/Korean word alignment is not implemented yet"
        );
        let words = text::words(transcript);
        if words.is_empty() {
            return Ok(Vec::new());
        }
        let tok = self
            .aligner_tokenizer
            .clone()
            .context("word timestamps require an aligner tokenizer")?;
        let network = self.session(Kind::Aligner)?;
        ensure!(
            network
                .config
                .support_languages
                .iter()
                .any(|v| v.eq_ignore_ascii_case(&language)),
            "aligner does not support {language}"
        );
        let timestamp = network
            .config
            .timestamp_token_id
            .context("aligner has no timestamp token")?;
        let tick = network
            .config
            .timestamp_segment_time
            .context("aligner has no timestamp resolution")?;
        let audio = network.audio(samples)?;
        let c = &network.config.thinker_config;
        let mut ids = vec![c.audio_start_token_id];
        let audio_start = ids.len();
        ids.extend(std::iter::repeat_n(c.audio_token_id, audio.dim(0)?));
        ids.push(c.audio_end_token_id);
        let mut positions = Vec::with_capacity(words.len() * 2);
        for word in &words {
            ids.extend(text::encode(&tok, word)?);
            positions.push(ids.len());
            ids.push(timestamp);
            positions.push(ids.len());
            ids.push(timestamp);
        }
        let x = splice(network, &ids, audio_start, &audio)?;
        let logits = network.decode(&x, 0, &mut network.cache(), false)?;
        let mut timestamps = Vec::with_capacity(positions.len());
        for pos in positions {
            timestamps
                .push(u64::from(logits.i((0, pos, ..))?.argmax(0)?.to_scalar::<u32>()?) * tick);
        }
        let timestamps = text::repair_timestamps(&timestamps);
        let duration = (samples.len() as u64 * 1000).div_ceil(u64::from(Audio::SAMPLE_RATE));
        Ok(words
            .into_iter()
            .enumerate()
            .map(|(i, text)| Word {
                text,
                start_ms: timestamps[2 * i].min(duration),
                end_ms: timestamps[2 * i + 1].min(duration),
            })
            .collect())
    }
}

fn splice(network: &Network, ids: &[u32], audio_start: usize, audio: &Tensor) -> Result<Tensor> {
    let n = audio.dim(0)?;
    let x = network.embed(ids)?;
    Ok(Tensor::cat(
        &[
            x.narrow(1, 0, audio_start)?,
            audio.unsqueeze(0)?,
            x.narrow(1, audio_start + n, ids.len() - audio_start - n)?,
        ],
        1,
    )?)
}

impl AsrModel for Qwen3 {
    fn info(&self) -> ModelInfo {
        ModelInfo {
            id: self.id.clone(),
            family: "qwen3-asr".into(),
            word_timestamps: self.aligner_dir.is_some(),
        }
    }
    fn transcribe(&mut self, audio: &Audio, options: &TranscribeOptions) -> Result<Transcript> {
        ensure!(
            (1..=30).contains(&options.chunk_seconds),
            "chunk_seconds must be 1..=30"
        );
        ensure!(
            options.max_new_tokens > 0,
            "max_new_tokens must be positive"
        );
        if options.timestamps == TimestampMode::Word {
            ensure!(
                self.aligner_dir.is_some(),
                "word timestamps require Qwen3-ForcedAligner"
            );
        }
        let chunk_size = options.chunk_seconds as usize * Audio::SAMPLE_RATE as usize;
        let mut segments = Vec::new();
        for (index, samples) in audio.samples().chunks(chunk_size).enumerate() {
            let (text, language) = self
                .transcribe_chunk(samples, options)
                .with_context(|| format!("transcribe chunk {index}"))?;
            let start_ms = index as u64 * u64::from(options.chunk_seconds) * 1000;
            let end_ms = (start_ms
                + (samples.len() as u64 * 1000).div_ceil(u64::from(Audio::SAMPLE_RATE)))
            .min(audio.duration_ms());
            segments.push(Segment {
                text,
                language,
                start_ms,
                end_ms,
                words: Vec::new(),
            });
        }
        // Run all ASR chunks before switching to the aligner, so each set of
        // weights is loaded only once per task, regardless of audio duration.
        if options.timestamps == TimestampMode::Word {
            for (index, segment) in segments.iter_mut().enumerate() {
                let samples = &audio.samples()
                    [index * chunk_size..((index + 1) * chunk_size).min(audio.samples().len())];
                segment.words = self
                    .align_chunk(samples, &segment.text, &segment.language)
                    .with_context(|| format!("align chunk {index}"))?;
                for word in &mut segment.words {
                    word.start_ms += segment.start_ms;
                    word.end_ms += segment.start_ms;
                }
            }
        }
        let language = segments
            .iter()
            .find(|s| !s.text.is_empty())
            .or_else(|| segments.first())
            .map(|s| s.language.clone())
            .unwrap_or_default();
        let separator = if ["Chinese", "Cantonese", "Japanese"].contains(&language.as_str()) {
            ""
        } else {
            " "
        };
        let text = segments
            .iter()
            .map(|s| s.text.as_str())
            .filter(|t| !t.is_empty())
            .collect::<Vec<_>>()
            .join(separator);
        Ok(Transcript {
            model: self.id.clone(),
            text,
            language,
            duration_ms: audio.duration_ms(),
            segments,
        })
    }
}
