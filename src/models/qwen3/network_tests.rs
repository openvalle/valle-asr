use super::*;
use std::collections::HashMap;

fn model() -> Result<Network> {
    let config: Config = serde_json::from_value(serde_json::json!({
        "thinker_config": {
            "audio_start_token_id": 1, "audio_end_token_id": 2, "audio_token_id": 3,
            "classify_num": 5,
            "audio_config": {
                "d_model": 4, "encoder_layers": 1, "encoder_attention_heads": 2,
                "encoder_ffn_dim": 6, "num_mel_bins": 128, "max_source_positions": 32,
                "n_window": 50, "n_window_infer": 100, "downsample_hidden_size": 1,
                "output_dim": 4
            },
            "text_config": {
                "hidden_size": 4, "intermediate_size": 6, "num_hidden_layers": 2,
                "num_attention_heads": 2, "num_key_value_heads": 1, "head_dim": 2,
                "rms_norm_eps": 0.000001, "rope_theta": 10000.0, "vocab_size": 7,
                "max_position_embeddings": 64, "tie_word_embeddings": false
            }
        }
    }))?;
    config.validate()?;
    let mut weights = HashMap::new();
    for (name, rows, cols) in [
        ("self_attn.q_proj", 4, 4),
        ("self_attn.k_proj", 2, 4),
        ("self_attn.v_proj", 2, 4),
        ("self_attn.o_proj", 4, 4),
        ("mlp.gate_proj", 6, 4),
        ("mlp.up_proj", 6, 4),
        ("mlp.down_proj", 4, 6),
    ] {
        let values: Vec<_> = (0..rows * cols)
            .map(|index| ((index % 7) as f32 - 3.0) * 0.05)
            .collect();
        weights.insert(
            format!("{name}.weight"),
            Tensor::from_vec(values, (rows, cols), &Device::Cpu)?,
        );
    }
    for (name, width) in [
        ("input_layernorm", 4),
        ("post_attention_layernorm", 4),
        ("self_attn.q_norm", 2),
        ("self_attn.k_norm", 2),
        ("norm", 4),
    ] {
        weights.insert(
            format!("{name}.weight"),
            Tensor::ones(width, DType::F32, &Device::Cpu)?,
        );
    }
    let vb = VarBuilder::from_tensors(weights, DType::F32, &Device::Cpu);
    let c = &config.thinker_config;
    Ok(Network {
        encoder: Encoder::load(VarBuilder::zeros(DType::F32, &Device::Cpu), &c.audio_config)?,
        embedding: Tensor::zeros((7, 4), DType::F32, &Device::Cpu)?,
        layers: (0..2)
            .map(|_| TextLayer::load(vb.clone(), &c.text_config))
            .collect::<Result<_>>()?,
        norm: rms_norm(4, 0.000001, vb.pp("norm"))?,
        head: Linear::new(
            Tensor::from_vec(
                (0..20).map(|i| (i as f32 - 10.0) * 0.1).collect(),
                (5, 4),
                &Device::Cpu,
            )?,
            None,
        ),
        config,
    })
}

fn input() -> Result<Tensor> {
    Ok(Tensor::from_vec(
        vec![
            0.2f32, -0.8, 1.0, 0.4, 0.7, 0.1, -0.3, 0.9, -0.5, 0.4, 0.8, 0.2,
        ],
        (1, 3, 4),
        &Device::Cpu,
    )?)
}

fn close(actual: &Tensor, expected: &Tensor) -> Result<()> {
    let actual = actual.flatten_all()?.to_vec1::<f32>()?;
    let expected = expected.flatten_all()?.to_vec1::<f32>()?;
    assert!(expected.iter().any(|x| x.abs() > 0.01));
    assert_eq!(actual.len(), expected.len());
    for (a, b) in actual.into_iter().zip(expected) {
        assert!((a - b).abs() < 1e-5, "{a} != {b}");
    }
    Ok(())
}

#[test]
fn timestamp_rows_match_full_context_classifier() -> Result<()> {
    let model = model()?;
    let input = input()?;
    let complete = model.decode(&input, 0, &mut model.cache(), false)?;
    // Out-of-order, repeated positions ensure gathering preserves row identity.
    let selected = model.classify_positions(&input, &[2, 0, 2])?;
    for (row, position) in [2, 0, 2].into_iter().enumerate() {
        close(&selected.i((0, row, ..))?, &complete.i((0, position, ..))?)?;
    }
    Ok(())
}

#[test]
fn incremental_decode_matches_causal_full_forward() -> Result<()> {
    let model = model()?;
    let input = input()?;
    let complete = model.decode(&input, 0, &mut model.cache(), false)?;
    let mut cache = model.cache();
    for offset in 0..3 {
        let step = model.decode(&input.narrow(1, offset, 1)?, offset, &mut cache, true)?;
        close(&step, &complete.i((.., offset..offset + 1, ..))?)?;
    }
    Ok(())
}
