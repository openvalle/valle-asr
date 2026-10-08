//! Native Candle inference, following the official Apache-2.0 Qwen architecture.
//! See THIRD_PARTY.md for the pinned architecture and Rust reference sources.
use super::config::{AudioConfig, Config, TextConfig};
use anyhow::{Context, Result, ensure};
use candle_core::{DType, Device, IndexOp, Tensor};
use candle_nn::{
    Conv2d, Conv2dConfig, LayerNorm, Linear, Module, RmsNorm, VarBuilder, layer_norm, linear,
    linear_no_bias, rms_norm,
};
use std::{collections::BTreeSet, path::Path};

fn attention(q: &Tensor, k: &Tensor, v: &Tensor, mask: Option<&Tensor>) -> Result<Tensor> {
    let mut scores = (q.matmul(&k.transpose(2, 3)?.contiguous()?)? / (q.dim(3)? as f64).sqrt())?;
    if let Some(mask) = mask {
        scores = scores.broadcast_add(mask)?;
    }
    Ok(candle_nn::ops::softmax_last_dim(&scores)?.matmul(v)?)
}

struct AudioLayer {
    norm1: LayerNorm,
    norm2: LayerNorm,
    q: Linear,
    k: Linear,
    v: Linear,
    out: Linear,
    fc1: Linear,
    fc2: Linear,
    heads: usize,
}
impl AudioLayer {
    fn load(vb: VarBuilder<'_>, c: &AudioConfig) -> Result<Self> {
        let d = c.d_model;
        let a = vb.pp("self_attn");
        Ok(Self {
            norm1: layer_norm(d, 1e-5, vb.pp("self_attn_layer_norm"))?,
            norm2: layer_norm(d, 1e-5, vb.pp("final_layer_norm"))?,
            q: linear(d, d, a.pp("q_proj"))?,
            k: linear(d, d, a.pp("k_proj"))?,
            v: linear(d, d, a.pp("v_proj"))?,
            out: linear(d, d, a.pp("out_proj"))?,
            fc1: linear(d, c.encoder_ffn_dim, vb.pp("fc1"))?,
            fc2: linear(c.encoder_ffn_dim, d, vb.pp("fc2"))?,
            heads: c.encoder_attention_heads,
        })
    }
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (b, s, d) = x.dims3()?;
        let n = self.norm1.forward(x)?;
        let project = |p: &Linear| -> Result<Tensor> {
            Ok(p.forward(&n)?
                .reshape((b, s, self.heads, d / self.heads))?
                .transpose(1, 2)?
                .contiguous()?)
        };
        let h = attention(
            &project(&self.q)?,
            &project(&self.k)?,
            &project(&self.v)?,
            None,
        )?;
        let h = h.transpose(1, 2)?.contiguous()?.reshape((b, s, d))?;
        let x = (x + self.out.forward(&h)?)?;
        Ok((&x
            + self
                .fc2
                .forward(&self.fc1.forward(&self.norm2.forward(&x)?)?.gelu_erf()?)?)?)
    }
}

struct Encoder {
    conv: [Conv2d; 3],
    projection: Linear,
    layers: Vec<AudioLayer>,
    norm: LayerNorm,
    proj1: Linear,
    proj2: Linear,
    positions: Tensor,
    window_frames: usize,
}
impl Encoder {
    fn load(vb: VarBuilder<'_>, c: &AudioConfig) -> Result<Self> {
        let channels = c.downsample_hidden_size;
        let make_conv = |name: &str, input: usize| -> Result<Conv2d> {
            let p = vb.pp(name);
            Ok(Conv2d::new(
                p.get((channels, input, 3, 3), "weight")?,
                Some(p.get(channels, "bias")?),
                Conv2dConfig {
                    padding: 1,
                    stride: 2,
                    ..Default::default()
                },
            ))
        };
        let mut data = vec![0.0f32; c.max_source_positions * c.d_model];
        let half = c.d_model / 2;
        for pos in 0..c.max_source_positions {
            for j in 0..half {
                let phase = pos as f64 * (-(j as f64) * 10000f64.ln() / (half - 1) as f64).exp();
                data[pos * c.d_model + j] = phase.sin() as f32;
                data[pos * c.d_model + half + j] = phase.cos() as f32;
            }
        }
        Ok(Self {
            conv: [
                make_conv("conv2d1", 1)?,
                make_conv("conv2d2", channels)?,
                make_conv("conv2d3", channels)?,
            ],
            projection: linear_no_bias(
                channels * c.num_mel_bins.div_ceil(8),
                c.d_model,
                vb.pp("conv_out"),
            )?,
            layers: (0..c.encoder_layers)
                .map(|i| AudioLayer::load(vb.pp(format!("layers.{i}")), c))
                .collect::<Result<_>>()?,
            norm: layer_norm(c.d_model, 1e-5, vb.pp("ln_post"))?,
            proj1: linear(c.d_model, c.d_model, vb.pp("proj1"))?,
            proj2: linear(c.d_model, c.output_dim, vb.pp("proj2"))?,
            positions: Tensor::from_vec(data, (c.max_source_positions, c.d_model), &Device::Cpu)?,
            window_frames: c.n_window_infer,
        })
    }
    fn forward(&self, mel: &Tensor) -> Result<Tensor> {
        let frames = mel.dim(1)?;
        let mut outputs = Vec::new();
        // Each attention window is independent. Execute it separately to bound RAM.
        for start in (0..frames).step_by(self.window_frames) {
            let end = (start + self.window_frames).min(frames);
            let mut stems = Vec::new();
            for chunk in (start..end).step_by(100) {
                let count = (end - chunk).min(100);
                let part = mel.narrow(1, chunk, count)?;
                let pad = Tensor::zeros((128, 100 - count), DType::F32, &Device::Cpu)?;
                let mut x = Tensor::cat(&[part, pad], 1)?.unsqueeze(0)?.unsqueeze(0)?;
                for conv in &self.conv {
                    x = conv.forward(&x)?.gelu_erf()?;
                }
                let (b, c, f, t) = x.dims4()?;
                let x = x
                    .permute((0, 3, 1, 2))?
                    .contiguous()?
                    .reshape((b, t, c * f))?;
                let x = self
                    .projection
                    .forward(&x)?
                    .broadcast_add(&self.positions.narrow(0, 0, t)?.unsqueeze(0)?)?;
                stems.push(x.narrow(1, 0, count.div_ceil(8))?);
            }
            let mut x = Tensor::cat(&stems, 1)?;
            for layer in &self.layers {
                x = layer.forward(&x)?;
            }
            let x = self
                .proj2
                .forward(&self.proj1.forward(&self.norm.forward(&x)?)?.gelu_erf()?)?;
            outputs.push(x.squeeze(0)?);
        }
        Ok(Tensor::cat(&outputs, 0)?)
    }
}

struct TextLayer {
    norm1: RmsNorm,
    norm2: RmsNorm,
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    q: Linear,
    k: Linear,
    v: Linear,
    out: Linear,
    gate: Linear,
    up: Linear,
    down: Linear,
    heads: usize,
    kv_heads: usize,
    head_dim: usize,
}
type Cache = Vec<Option<(Tensor, Tensor)>>;
impl TextLayer {
    fn load(vb: VarBuilder<'_>, c: &TextConfig) -> Result<Self> {
        let d = c.hidden_size;
        let a = vb.pp("self_attn");
        let m = vb.pp("mlp");
        Ok(Self {
            norm1: rms_norm(d, c.rms_norm_eps, vb.pp("input_layernorm"))?,
            norm2: rms_norm(d, c.rms_norm_eps, vb.pp("post_attention_layernorm"))?,
            q_norm: rms_norm(c.head_dim, c.rms_norm_eps, a.pp("q_norm"))?,
            k_norm: rms_norm(c.head_dim, c.rms_norm_eps, a.pp("k_norm"))?,
            q: linear_no_bias(d, c.num_attention_heads * c.head_dim, a.pp("q_proj"))?,
            k: linear_no_bias(d, c.num_key_value_heads * c.head_dim, a.pp("k_proj"))?,
            v: linear_no_bias(d, c.num_key_value_heads * c.head_dim, a.pp("v_proj"))?,
            out: linear_no_bias(c.num_attention_heads * c.head_dim, d, a.pp("o_proj"))?,
            gate: linear_no_bias(d, c.intermediate_size, m.pp("gate_proj"))?,
            up: linear_no_bias(d, c.intermediate_size, m.pp("up_proj"))?,
            down: linear_no_bias(c.intermediate_size, d, m.pp("down_proj"))?,
            heads: c.num_attention_heads,
            kv_heads: c.num_key_value_heads,
            head_dim: c.head_dim,
        })
    }
    fn forward(
        &self,
        x: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        cache: &mut Option<(Tensor, Tensor)>,
        mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let (b, s, _) = x.dims3()?;
        let n = self.norm1.forward(x)?;
        let project = |p: &Linear, h: usize| -> Result<Tensor> {
            Ok(p.forward(&n)?
                .reshape((b, s, h, self.head_dim))?
                .transpose(1, 2)?
                .contiguous()?)
        };
        let q = rotary(
            &self.q_norm.forward(&project(&self.q, self.heads)?)?,
            cos,
            sin,
        )?;
        let k = rotary(
            &self.k_norm.forward(&project(&self.k, self.kv_heads)?)?,
            cos,
            sin,
        )?;
        let v = project(&self.v, self.kv_heads)?;
        let (k, v) = if let Some((pk, pv)) = cache.as_ref() {
            (Tensor::cat(&[pk, &k], 2)?, Tensor::cat(&[pv, &v], 2)?)
        } else {
            (k, v)
        };
        *cache = Some((k.clone(), v.clone()));
        let repeats = self.heads / self.kv_heads;
        let repeat = |t: Tensor| -> Result<Tensor> {
            let seq = t.dim(2)?;
            Ok(t.unsqueeze(2)?
                .expand((b, self.kv_heads, repeats, seq, self.head_dim))?
                .reshape((b, self.heads, seq, self.head_dim))?
                .contiguous()?)
        };
        let h = attention(&q, &repeat(k)?, &repeat(v)?, mask)?
            .transpose(1, 2)?
            .contiguous()?
            .reshape((b, s, self.heads * self.head_dim))?;
        let x = (x + self.out.forward(&h)?)?;
        let n = self.norm2.forward(&x)?;
        Ok((&x
            + self
                .down
                .forward(&(self.gate.forward(&n)?.silu()? * self.up.forward(&n)?)?)?)?)
    }
}

fn rotary(x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    let half = x.dim(3)? / 2;
    let rotated = Tensor::cat(
        &[(x.narrow(3, half, half)? * -1.0)?, x.narrow(3, 0, half)?],
        3,
    )?;
    Ok((x.broadcast_mul(cos)? + rotated.broadcast_mul(sin)?)?)
}

pub(super) struct Network {
    pub config: Config,
    encoder: Encoder,
    embedding: Tensor,
    layers: Vec<TextLayer>,
    norm: RmsNorm,
    head: Linear,
}
impl Network {
    pub fn load(root: &Path) -> Result<Self> {
        let config: Config = serde_json::from_slice(&std::fs::read(root.join("config.json"))?)?;
        config.validate()?;
        let index = root.join("model.safetensors.index.json");
        let paths = if index.exists() {
            let d: serde_json::Value = serde_json::from_slice(&std::fs::read(index)?)?;
            let map = d["weight_map"]
                .as_object()
                .context("invalid safetensors shard index")?;
            let mut names = BTreeSet::new();
            for name in map.values() {
                let name = name.as_str().context("invalid shard filename")?;
                ensure!(
                    Path::new(name).file_name().and_then(|s| s.to_str()) == Some(name),
                    "unsafe shard filename"
                );
                names.insert(root.join(name));
            }
            names.into_iter().collect::<Vec<_>>()
        } else {
            vec![root.join("model.safetensors")]
        };
        // Candle uses portable memmap2. The model cache is immutable while open;
        // updates are downloaded to another file and atomically renamed.
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&paths, DType::F32, &Device::Cpu)? };
        let t = &config.thinker_config.text_config;
        let text = vb.pp("thinker.model");
        let embedding = text.get((t.vocab_size, t.hidden_size), "embed_tokens.weight")?;
        let head = if t.tie_word_embeddings && config.thinker_config.classify_num.is_none() {
            Linear::new(embedding.clone(), None)
        } else {
            linear_no_bias(
                t.hidden_size,
                config.thinker_config.classify_num.unwrap_or(t.vocab_size),
                vb.pp("thinker.lm_head"),
            )?
        };
        Ok(Self {
            encoder: Encoder::load(
                vb.pp("thinker.audio_tower"),
                &config.thinker_config.audio_config,
            )?,
            embedding,
            layers: (0..t.num_hidden_layers)
                .map(|i| TextLayer::load(text.pp(format!("layers.{i}")), t))
                .collect::<Result<_>>()?,
            norm: rms_norm(t.hidden_size, t.rms_norm_eps, text.pp("norm"))?,
            head,
            config,
        })
    }
    pub fn audio(&self, samples: &[f32]) -> Result<Tensor> {
        self.encoder.forward(&super::dsp::mel(samples)?)
    }
    pub fn embed(&self, ids: &[u32]) -> Result<Tensor> {
        Ok(self
            .embedding
            .index_select(&Tensor::new(ids, &Device::Cpu)?, 0)?
            .unsqueeze(0)?)
    }
    pub fn cache(&self) -> Cache {
        vec![None; self.layers.len()]
    }
    pub fn decode(
        &self,
        x: &Tensor,
        offset: usize,
        cache: &mut Cache,
        last_only: bool,
    ) -> Result<Tensor> {
        let seq = x.dim(1)?;
        let t = &self.config.thinker_config.text_config;
        ensure!(
            offset + seq <= t.max_position_embeddings,
            "decoder context limit exceeded"
        );
        // Audio uses one-dimensional positions on all three MRoPE axes. The
        // resulting cos/sin are ordinary RoPE, including interleaved MRoPE.
        let mut cos = Vec::with_capacity(seq * t.head_dim);
        let mut sin = Vec::with_capacity(seq * t.head_dim);
        for pos in offset..offset + seq {
            for _ in 0..2 {
                for i in 0..t.head_dim / 2 {
                    let phase = pos as f64 / t.rope_theta.powf(2.0 * i as f64 / t.head_dim as f64);
                    cos.push(phase.cos() as f32);
                    sin.push(phase.sin() as f32);
                }
            }
        }
        let shape = (1, 1, seq, t.head_dim);
        let cos = Tensor::from_vec(cos, shape, &Device::Cpu)?;
        let sin = Tensor::from_vec(sin, shape, &Device::Cpu)?;
        let mask = if seq > 1 {
            let total = seq + offset;
            let data = (0..seq)
                .flat_map(|i| {
                    (0..total).map(move |j| {
                        if j > offset + i {
                            f32::NEG_INFINITY
                        } else {
                            0.0
                        }
                    })
                })
                .collect::<Vec<_>>();
            Some(Tensor::from_vec(data, (1, 1, seq, total), &Device::Cpu)?)
        } else {
            None
        };
        let mut h = x.clone();
        for (layer, kv) in self.layers.iter().zip(cache.iter_mut()) {
            h = layer.forward(&h, &cos, &sin, kv, mask.as_ref())?;
        }
        let h = if last_only {
            h.i((.., seq - 1..seq, ..))?
        } else {
            h
        };
        Ok(self.head.forward(&self.norm.forward(&h)?)?)
    }
}
