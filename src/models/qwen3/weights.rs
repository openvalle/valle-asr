//! Convert mapped BF16 weights straight into their final CPU F32 storage.
use candle_core::{DType, Device, Error, Result, Shape, Tensor, safetensors::MmapedSafetensors};
use candle_nn::{Init, VarBuilder, var_builder::SimpleBackend};
use std::path::PathBuf;

struct CpuWeights(MmapedSafetensors);

impl CpuWeights {
    fn tensor(&self, name: &str, dtype: DType, device: &Device) -> Result<Tensor> {
        let view = self.0.get(name)?;
        if matches!(device, Device::Cpu)
            && dtype == DType::F32
            && DType::try_from(view.dtype())? == DType::BF16
        {
            // Safetensors stores little-endian data. Reading bytes avoids
            // alignment assumptions and an intermediate owned BF16 tensor.
            let values: Vec<f32> = view
                .data()
                .as_chunks::<2>()
                .0
                .iter()
                .map(|bytes| {
                    let bits = u16::from_le_bytes([bytes[0], bytes[1]]);
                    // Match half/Candle's quiet-NaN conversion as well.
                    let bits = if bits & 0x7fff > 0x7f80 {
                        bits | 0x0040
                    } else {
                        bits
                    };
                    f32::from_bits(u32::from(bits) << 16)
                })
                .collect();
            Tensor::from_vec(values, view.shape(), device)
        } else {
            self.0.load(name, device)?.to_dtype(dtype)
        }
    }
}

impl SimpleBackend for CpuWeights {
    fn get(
        &self,
        shape: Shape,
        name: &str,
        _: Init,
        dtype: DType,
        device: &Device,
    ) -> Result<Tensor> {
        let actual: Shape = self.0.get(name)?.shape().into();
        if actual != shape {
            return Err(Error::UnexpectedShape {
                msg: format!("shape mismatch for {name}"),
                expected: shape,
                got: actual,
            }
            .bt());
        }
        self.tensor(name, dtype, device)
    }

    fn get_unchecked(&self, name: &str, dtype: DType, device: &Device) -> Result<Tensor> {
        self.tensor(name, dtype, device)
    }

    fn contains_tensor(&self, name: &str) -> bool {
        self.0.get(name).is_ok()
    }
}

pub(super) fn load(paths: &[PathBuf]) -> Result<VarBuilder<'static>> {
    // SAFETY: model files are immutable while mapped. Cache updates use a
    // separate file and atomic rename; returned tensors own their final data.
    let mapped = unsafe { MmapedSafetensors::multi(paths)? };
    Ok(VarBuilder::from_backend(
        Box::new(CpuWeights(mapped)),
        DType::F32,
        Device::Cpu,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn safetensor(
        name: &str,
        dtype: &str,
        shape: &[usize],
        data: &[u8],
    ) -> tempfile::NamedTempFile {
        let mut file = tempfile::NamedTempFile::new().unwrap();
        let mut header = serde_json::to_vec(&serde_json::json!({
            name: {"dtype": dtype, "shape": shape, "data_offsets": [0, data.len()]}
        }))
        .unwrap();
        header.resize(header.len().next_multiple_of(8), b' ');
        file.write_all(&(header.len() as u64).to_le_bytes())
            .unwrap();
        file.write_all(&header).unwrap();
        file.write_all(data).unwrap();
        file.flush().unwrap();
        file
    }

    #[test]
    fn bf16_matches_candle_for_every_bit_pattern() -> Result<()> {
        let data: Vec<_> = (0..=u16::MAX).flat_map(u16::to_le_bytes).collect();
        let file = safetensor("weight", "BF16", &[256, 256], &data);
        let paths = [file.path().to_path_buf()];
        let fused = load(&paths)?.get((256, 256), "weight")?;
        // Independent existing loader verifies zeros, subnormals, infinities,
        // NaNs and normal values, not just the values in one model checkpoint.
        let reference = unsafe { MmapedSafetensors::multi(&paths)? }
            .load("weight", &Device::Cpu)?
            .to_dtype(DType::F32)?;
        for (actual, expected) in fused
            .flatten_all()?
            .to_vec1::<f32>()?
            .into_iter()
            .zip(reference.flatten_all()?.to_vec1::<f32>()?)
        {
            assert_eq!(actual.to_bits(), expected.to_bits());
        }
        Ok(())
    }

    #[test]
    fn shards_shapes_and_f32_fallback() -> Result<()> {
        let first = safetensor("block.weight", "BF16", &[2], &[0x80, 0x3f, 0x00, 0xc0]);
        let data: Vec<_> = [3.5f32, -0.25]
            .into_iter()
            .flat_map(f32::to_le_bytes)
            .collect();
        let second = safetensor("other", "F32", &[2], &data);
        let vb = load(&[first.path().to_path_buf(), second.path().to_path_buf()])?;
        assert_eq!(
            vb.pp("block").get(2, "weight")?.to_vec1::<f32>()?,
            [1.0, -2.0]
        );
        assert_eq!(vb.get(2, "other")?.to_vec1::<f32>()?, [3.5, -0.25]);
        assert!(vb.get(3, "other").is_err());
        assert!(vb.get(2, "missing").is_err());
        assert!(vb.contains_tensor("other"));
        assert!(!vb.contains_tensor("missing"));
        assert_eq!(vb.get_unchecked("other")?.dims(), [2]);
        Ok(())
    }
}
