//! Qwen3-ASR and its separate learned timestamp classifier, running on CPU.
mod config;
mod dsp;
mod file;
mod network;
mod text;
mod weights;

use crate::{
    AsrModel, Audio, CancellationToken, ModelInfo, Segment, TimestampMode, TranscribeOptions,
    Transcript, Word,
};
use anyhow::{Context, Result, bail, ensure};
use candle_core::Tensor;
use network::Network;
use std::{
    path::{Path, PathBuf},
    sync::Arc,
};
use tokenizers::Tokenizer;

#[derive(Clone, Copy, PartialEq, Eq)]
enum Kind {
    Asr,
    Aligner,
}

/// Qwen3-ASR with optional learned word alignment and one resident CPU network.
/// ASR runs for all chunks first,
/// then its weights are released before the aligner loads. The final session
/// stays cached for repeated text-only transcription or standalone alignment.
pub struct Qwen3 {
    id: String,
    model_dir: PathBuf,
    aligner_dir: Option<PathBuf>,
    asr_tokenizer: Arc<Tokenizer>,
    aligner_tokenizer: Option<Arc<Tokenizer>>,
    resident: Option<(Kind, Network)>,
}

impl Qwen3 {
    /// Load configuration/tokenizers from official local checkpoint directories.
    /// Neural weights load lazily on the first inference request. The caller
    /// supplies an engine ID; it does not select a checkpoint architecture.
    ///
    /// # Errors
    /// Rejects unreadable or unsupported configuration/tokenizers, including
    /// an aligner checkpoint passed as the ASR directory. No files are downloaded.
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
            .map(|path| text::tokenizer(path).map(Arc::new))
            .transpose()?;
        Ok(Self {
            id: id.into(),
            asr_tokenizer: Arc::new(text::tokenizer(&model_dir)?),
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

    fn session(&mut self, kind: Kind, cancellation: &CancellationToken) -> Result<&Network> {
        cancellation.check()?;
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
            let loaded = Network::load(path, cancellation);
            // Candle weight errors erase their inner type; preserve our public
            // cancellation marker when a loading checkpoint aborted the request.
            cancellation.check()?;
            let network = loaded.with_context(|| format!("load {}", path.display()))?;
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
        options.cancellation.check()?;
        let tok = Arc::clone(&self.asr_tokenizer);
        let network = self.session(Kind::Asr, &options.cancellation)?;
        let audio = network.audio(samples, &options.cancellation)?;
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
        let mut logits = network.decode(&x, 0, &mut cache, true, &options.cancellation)?;
        let mut output = Vec::new();
        for step in 0..options.max_new_tokens {
            options.cancellation.check()?;
            let next = logits.flatten_all()?.argmax(0)?.to_scalar::<u32>()?;
            if [151643, 151645].contains(&next) {
                let decoded = tok
                    .decode(&output, false)
                    .map_err(|e| anyhow::anyhow!("decode: {e}"))?;
                return text::parse_output(&decoded, options.language.as_deref());
            }
            output.push(next);
            if step + 1 < options.max_new_tokens {
                logits = network.decode(
                    &network.embed(&[next])?,
                    ids.len() + step,
                    &mut cache,
                    true,
                    &options.cancellation,
                )?;
            }
        }
        bail!(
            "Qwen3 decode reached max_new_tokens without an end token; increase the limit or shorten chunks"
        )
    }

    /// Align a known transcript to a single clip of at most 30 seconds.
    ///
    /// Empty text returns no words. Times are relative to the supplied clip.
    /// Chinese/Cantonese units are Han characters.
    ///
    /// # Errors
    /// Rejects clips over 30 seconds, unsupported alignment languages, missing
    /// aligner weights, and failed model loading or inference.
    pub fn align(&mut self, audio: &Audio, transcript: &str, language: &str) -> Result<Vec<Word>> {
        self.align_with_cancellation(audio, transcript, language, &CancellationToken::default())
    }

    /// Align a known transcript with cooperative cancellation, including model loading.
    /// Checks occur between preprocessing steps and encoder/classifier layers.
    ///
    /// # Errors
    /// Returns [`crate::Cancelled`] on cancellation, or the same validation,
    /// loading and inference errors as [`Self::align`].
    pub fn align_with_cancellation(
        &mut self,
        audio: &Audio,
        transcript: &str,
        language: &str,
        cancellation: &CancellationToken,
    ) -> Result<Vec<Word>> {
        cancellation.check()?;
        ensure!(
            audio.samples().len() <= 30 * Audio::SAMPLE_RATE as usize,
            "alignment clips must be at most 30 seconds"
        );
        let words = self.align_chunk(audio.samples(), transcript, language, cancellation)?;
        cancellation.check()?;
        Ok(words)
    }

    fn align_chunk(
        &mut self,
        samples: &[f32],
        transcript: &str,
        language: &str,
        cancellation: &CancellationToken,
    ) -> Result<Vec<Word>> {
        cancellation.check()?;
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
        let network = self.session(Kind::Aligner, cancellation)?;
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
        let audio = network.audio(samples, cancellation)?;
        let c = &network.config.thinker_config;
        let mut ids = vec![c.audio_start_token_id];
        let audio_start = ids.len();
        ids.extend(std::iter::repeat_n(c.audio_token_id, audio.dim(0)?));
        ids.push(c.audio_end_token_id);
        let mut positions = Vec::with_capacity(words.len() * 2);
        for word in &words {
            cancellation.check()?;
            ids.extend(text::encode(&tok, word)?);
            positions.push(u32::try_from(ids.len())?);
            ids.push(timestamp);
            positions.push(u32::try_from(ids.len())?);
            ids.push(timestamp);
        }
        let x = splice(network, &ids, audio_start, &audio)?;
        let timestamps: Vec<_> = network
            .classify_positions(&x, &positions, cancellation)?
            .squeeze(0)?
            .argmax(1)?
            .to_vec1::<u32>()?
            .into_iter()
            .map(|index| u64::from(index) * tick)
            .collect();
        cancellation.check()?;
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
    Ok(Tensor::cat(
        &[
            network.embed(&ids[..audio_start])?,
            audio.unsqueeze(0)?,
            network.embed(&ids[audio_start + n..])?,
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

    fn transcribe_file(
        &mut self,
        path: &Path,
        options: &TranscribeOptions,
        emit: &mut dyn FnMut(Segment) -> Result<()>,
    ) -> Result<crate::TranscriptSummary> {
        self.transcribe_wav(path, options, emit)
    }

    fn transcribe(&mut self, audio: &Audio, options: &TranscribeOptions) -> Result<Transcript> {
        options.validate()?;
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
                    .align_chunk(
                        samples,
                        &segment.text,
                        &segment.language,
                        &options.cancellation,
                    )
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
        options.cancellation.check()?;
        Ok(Transcript {
            model: self.id.clone(),
            text,
            language,
            duration_ms: audio.duration_ms(),
            segments,
        })
    }
}
