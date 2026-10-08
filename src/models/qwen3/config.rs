use anyhow::{Result, ensure};
use serde::Deserialize;

#[derive(Debug, Deserialize)]
pub(super) struct Config {
    pub thinker_config: Thinker,
    #[serde(default)]
    pub timestamp_token_id: Option<u32>,
    #[serde(default)]
    pub timestamp_segment_time: Option<u64>,
    #[serde(default)]
    pub support_languages: Vec<String>,
}

#[derive(Debug, Deserialize)]
pub(super) struct Thinker {
    pub audio_config: AudioConfig,
    pub text_config: TextConfig,
    pub audio_start_token_id: u32,
    pub audio_end_token_id: u32,
    pub audio_token_id: u32,
    #[serde(default)]
    pub classify_num: Option<usize>,
}

#[derive(Debug, Deserialize)]
pub(super) struct AudioConfig {
    pub d_model: usize,
    pub encoder_layers: usize,
    pub encoder_attention_heads: usize,
    pub encoder_ffn_dim: usize,
    pub num_mel_bins: usize,
    pub max_source_positions: usize,
    pub n_window: usize,
    pub n_window_infer: usize,
    pub downsample_hidden_size: usize,
    pub output_dim: usize,
}

#[derive(Debug, Deserialize)]
pub(super) struct TextConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub rms_norm_eps: f64,
    pub rope_theta: f64,
    pub vocab_size: usize,
    pub max_position_embeddings: usize,
    pub tie_word_embeddings: bool,
}

impl Config {
    pub fn validate(&self) -> Result<()> {
        let a = &self.thinker_config.audio_config;
        let t = &self.thinker_config.text_config;
        ensure!(
            a.num_mel_bins == 128,
            "Qwen3 frontend requires 128 mel bins"
        );
        ensure!(
            a.d_model > 0
                && a.encoder_attention_heads > 0
                && a.d_model.is_multiple_of(a.encoder_attention_heads),
            "invalid encoder dimensions"
        );
        ensure!(
            a.n_window == 50 && a.n_window_infer >= 100 && a.n_window_infer.is_multiple_of(100),
            "unsupported audio window layout"
        );
        ensure!(
            a.output_dim == t.hidden_size,
            "audio/text projection dimensions differ"
        );
        ensure!(
            t.num_key_value_heads > 0
                && t.num_attention_heads.is_multiple_of(t.num_key_value_heads)
                && t.head_dim > 0
                && t.head_dim.is_multiple_of(2),
            "invalid decoder dimensions"
        );
        ensure!(
            t.num_hidden_layers > 0 && t.max_position_embeddings > 0,
            "invalid decoder length"
        );
        ensure!(
            t.rope_theta.is_finite() && t.rope_theta > 0.0 && t.rms_norm_eps > 0.0,
            "invalid decoder normalization"
        );
        Ok(())
    }
}
