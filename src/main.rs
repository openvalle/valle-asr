use anyhow::{Result, bail};
use clap::{Parser, Subcommand};
use std::{
    io::BufWriter,
    path::{Path, PathBuf},
};

#[derive(Parser)]
#[command(
    version,
    about = "Local speech recognition with selectable models and word timestamps"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// List the built-in model catalog.
    Models,
    /// Download and verify a pinned ASR model and its aligner.
    Download {
        #[arg(long, default_value = "qwen3-asr-0.6b")]
        model: String,
        #[arg(long)]
        cache_dir: Option<PathBuf>,
        #[arg(long)]
        offline: bool,
    },
    /// Transcribe a WAV file. Word timestamps are enabled by default.
    Transcribe {
        input: PathBuf,
        #[arg(long, default_value = "qwen3-asr-0.6b")]
        model: String,
        #[arg(long)]
        model_dir: Option<PathBuf>,
        #[arg(long)]
        aligner_dir: Option<PathBuf>,
        #[arg(long)]
        cache_dir: Option<PathBuf>,
        #[arg(long)]
        offline: bool,
        #[arg(long)]
        language: Option<String>,
        #[arg(long)]
        text_only: bool,
        #[arg(long, default_value_t = 30)]
        chunk_seconds: u32,
        #[arg(long, default_value_t = 448)]
        max_new_tokens: usize,
        #[arg(long, default_value = "")]
        context: String,
        #[arg(long)]
        output: Option<PathBuf>,
    },
}

#[cfg(feature = "download")]
fn cache(root: Option<PathBuf>) -> Result<valle_asr::cache::ModelCache> {
    Ok(valle_asr::cache::ModelCache::new(match root {
        Some(root) => root,
        None => valle_asr::cache::ModelCache::default_path()?,
    }))
}

fn main() -> Result<()> {
    match Cli::parse().command {
        Command::Models => {
            #[cfg(feature = "download")]
            println!("{}",serde_json::to_string_pretty(&valle_asr::cache::builtin_models().iter().map(|m|serde_json::json!({"id":m.id,"repository":m.repository,"revision":m.revision,"license":m.license})).collect::<Vec<_>>())?);
            #[cfg(not(feature = "download"))]
            bail!("model catalog requires the download feature");
        }
        Command::Download {
            model,
            cache_dir,
            offline,
        } => {
            #[cfg(feature = "download")]
            {
                let cache = cache(cache_dir)?;
                for id in [&model, "qwen3-forced-aligner-0.6b"] {
                    let path = cache.ensure(&valle_asr::cache::builtin_model(id)?, offline)?;
                    println!("{}", path.display());
                }
            }
            #[cfg(not(feature = "download"))]
            {
                let _ = (model, cache_dir, offline);
                bail!("downloads are disabled");
            }
        }
        Command::Transcribe {
            input,
            model,
            model_dir,
            aligner_dir,
            cache_dir,
            offline,
            language,
            text_only,
            chunk_seconds,
            max_new_tokens,
            context,
            output,
        } => {
            #[cfg(feature = "qwen3")]
            {
                use valle_asr::{
                    AsrEngine, JsonTranscriptWriter, TimestampMode, TranscribeOptions,
                    models::qwen3::Qwen3,
                };
                if !model.starts_with("qwen3-asr-") {
                    bail!("unsupported model family: {model}");
                }
                #[cfg(feature = "download")]
                let (model_dir, aligner_dir) = {
                    let cache = cache(cache_dir)?;
                    let asr = match model_dir {
                        Some(p) => p,
                        None => cache.ensure(&valle_asr::cache::builtin_model(&model)?, offline)?,
                    };
                    let aligner = if text_only {
                        None
                    } else {
                        Some(match aligner_dir {
                            Some(p) => p,
                            None => cache.ensure(
                                &valle_asr::cache::builtin_model("qwen3-forced-aligner-0.6b")?,
                                offline,
                            )?,
                        })
                    };
                    (asr, aligner)
                };
                #[cfg(not(feature = "download"))]
                let (model_dir, aligner_dir) = {
                    let _ = (cache_dir, offline);
                    (
                        model_dir.ok_or_else(|| {
                            anyhow::anyhow!("--model-dir is required without downloads")
                        })?,
                        aligner_dir,
                    )
                };
                let mut engine = AsrEngine::new();
                engine.register(Qwen3::load(&model, model_dir, aligner_dir)?)?;
                let options = TranscribeOptions {
                    language,
                    timestamps: if text_only {
                        TimestampMode::None
                    } else {
                        TimestampMode::Word
                    },
                    chunk_seconds,
                    max_new_tokens,
                    context,
                    ..Default::default()
                };
                if let Some(output) = output {
                    let parent = output
                        .parent()
                        .filter(|p| !p.as_os_str().is_empty())
                        .unwrap_or(Path::new("."));
                    let mut temporary = tempfile::NamedTempFile::new_in(parent)?;
                    let mut writer =
                        JsonTranscriptWriter::new(BufWriter::new(temporary.as_file_mut()))?;
                    let summary =
                        engine.transcribe_file(&model, &input, &options, &mut |segment| {
                            writer.write_segment(segment)
                        })?;
                    writer.finish(&summary)?;
                    temporary.persist(&output)?;
                } else {
                    let stdout = std::io::stdout();
                    let mut writer = JsonTranscriptWriter::new(BufWriter::new(stdout.lock()))?;
                    let summary =
                        engine.transcribe_file(&model, &input, &options, &mut |segment| {
                            writer.write_segment(segment)
                        })?;
                    writer.finish(&summary)?;
                }
            }
            #[cfg(not(feature = "qwen3"))]
            {
                let _ = (
                    input,
                    model,
                    model_dir,
                    aligner_dir,
                    cache_dir,
                    offline,
                    language,
                    text_only,
                    chunk_seconds,
                    max_new_tokens,
                    context,
                    output,
                );
                bail!("Qwen3 backend is disabled");
            }
        }
    }
    Ok(())
}
