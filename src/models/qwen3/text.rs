use anyhow::{Context, Result, bail, ensure};
use std::path::Path;
use tokenizers::{
    AddedToken, SplitDelimiterBehavior, Tokenizer,
    decoders::byte_level::ByteLevel as ByteDecoder,
    models::bpe::BPE,
    pre_tokenizers::{
        byte_level::ByteLevel,
        sequence::Sequence,
        split::{Split, SplitPattern},
    },
};

pub(super) fn tokenizer(root: &Path) -> Result<Tokenizer> {
    let path = root.join("tokenizer.json");
    if path.exists() {
        return Tokenizer::from_file(path).map_err(|e| anyhow::anyhow!("tokenizer: {e}"));
    }
    let bpe = BPE::from_file(
        root.join("vocab.json")
            .to_str()
            .context("invalid tokenizer path")?,
        root.join("merges.txt")
            .to_str()
            .context("invalid tokenizer path")?,
    )
    .build()
    .map_err(|e| anyhow::anyhow!("BPE: {e}"))?;
    let mut tok = Tokenizer::new(bpe);
    let pattern = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";
    let split = Split::new(
        SplitPattern::Regex(pattern.into()),
        SplitDelimiterBehavior::Isolated,
        false,
    )
    .map_err(|e| anyhow::anyhow!("tokenizer split: {e}"))?;
    tok.with_pre_tokenizer(Some(Sequence::new(vec![
        split.into(),
        ByteLevel::new(false, true, false).into(),
    ])));
    tok.with_decoder(Some(ByteDecoder::default()));
    let config: serde_json::Value =
        serde_json::from_slice(&std::fs::read(root.join("tokenizer_config.json"))?)?;
    let mut added = config["added_tokens_decoder"]
        .as_object()
        .context("missing added tokens")?
        .iter()
        .map(|(id, token)| {
            Ok((
                id.parse::<u32>()?,
                token["content"]
                    .as_str()
                    .context("missing added token content")?
                    .to_owned(),
                token["special"].as_bool().unwrap_or(false),
            ))
        })
        .collect::<Result<Vec<_>>>()?;
    added.sort_by_key(|(id, _, _)| *id);
    for (id, content, special) in added {
        tok.add_tokens([AddedToken::from(content.clone(), special)])
            .map_err(|e| anyhow::anyhow!("added token: {e}"))?;
        ensure!(
            tok.token_to_id(&content) == Some(id),
            "token ID mismatch for {content}"
        );
    }
    Ok(tok)
}

pub(super) fn encode(tok: &Tokenizer, text: &str) -> Result<Vec<u32>> {
    Ok(tok
        .encode(text, false)
        .map_err(|e| anyhow::anyhow!("tokenize: {e}"))?
        .get_ids()
        .to_vec())
}

pub(super) fn language_name(language: &str) -> Result<String> {
    for (code, name) in [
        ("zh", "Chinese"),
        ("en", "English"),
        ("yue", "Cantonese"),
        ("ar", "Arabic"),
        ("de", "German"),
        ("fr", "French"),
        ("es", "Spanish"),
        ("pt", "Portuguese"),
        ("id", "Indonesian"),
        ("it", "Italian"),
        ("ko", "Korean"),
        ("ru", "Russian"),
        ("th", "Thai"),
        ("vi", "Vietnamese"),
        ("ja", "Japanese"),
        ("tr", "Turkish"),
        ("hi", "Hindi"),
        ("ms", "Malay"),
        ("nl", "Dutch"),
        ("sv", "Swedish"),
        ("da", "Danish"),
        ("fi", "Finnish"),
        ("pl", "Polish"),
        ("cs", "Czech"),
        ("fil", "Filipino"),
        ("fa", "Persian"),
        ("el", "Greek"),
        ("hu", "Hungarian"),
        ("mk", "Macedonian"),
        ("ro", "Romanian"),
    ] {
        if language.eq_ignore_ascii_case(code) || language.eq_ignore_ascii_case(name) {
            return Ok(name.into());
        }
    }
    bail!("unsupported language: {language}")
}

pub(super) fn parse_output(raw: &str, forced: Option<&str>) -> Result<(String, String)> {
    if let Some(language) = forced {
        return Ok((raw.trim().to_owned(), language_name(language)?));
    }
    let (header, text) = raw
        .split_once("<asr_text>")
        .context("Qwen3 output has no <asr_text> separator")?;
    let language = header
        .strip_prefix("language ")
        .context("Qwen3 output has no language header")?;
    if language.trim() == "None" {
        return Ok((text.trim().into(), "None".into()));
    }
    Ok((text.trim().into(), language_name(language.trim())?))
}

fn cjk(c: char) -> bool {
    matches!(c as u32,0x3400..=0x4dbf|0x4e00..=0x9fff|0xf900..=0xfaff|0x20000..=0x2ebef|0x30000..=0x323af)
}

/// The official processor uses one timestamp pair per Han character, and
/// whitespace-separated cleaned words for space-delimited languages.
pub(super) fn words(text: &str) -> Vec<String> {
    let mut result = Vec::new();
    for segment in text.split_whitespace() {
        let mut buf = String::new();
        for c in segment
            .chars()
            .filter(|c| c.is_alphanumeric() || *c == '\'')
        {
            if cjk(c) {
                if !buf.is_empty() {
                    result.push(std::mem::take(&mut buf));
                }
                result.push(c.to_string());
            } else {
                buf.push(c);
            }
        }
        if !buf.is_empty() {
            result.push(buf);
        }
    }
    result
}

/// Port of Qwen3ForceAlignProcessor.fix_timestamp (Apache-2.0): preserve the
/// longest nondecreasing subsequence, then repair outliers between anchors.
pub(super) fn repair_timestamps(input: &[u64]) -> Vec<u64> {
    if input.is_empty() {
        return Vec::new();
    }
    let n = input.len();
    let mut lengths = vec![1; n];
    let mut parents = vec![None; n];
    for i in 1..n {
        for j in 0..i {
            if input[j] <= input[i] && lengths[j] + 1 > lengths[i] {
                lengths[i] = lengths[j] + 1;
                parents[i] = Some(j);
            }
        }
    }
    let max = *lengths.iter().max().unwrap();
    let mut index = Some(lengths.iter().position(|v| *v == max).unwrap());
    let mut normal = vec![false; n];
    while let Some(i) = index {
        normal[i] = true;
        index = parents[i];
    }
    let mut result = input.to_vec();
    let mut i = 0;
    while i < n {
        if normal[i] {
            i += 1;
            continue;
        }
        let start = i;
        while i < n && !normal[i] {
            i += 1;
        }
        let left = start.checked_sub(1).map(|k| result[k]);
        let right = (i < n).then(|| result[i]);
        for (j, item) in result.iter_mut().enumerate().take(i).skip(start) {
            *item = match (left, right) {
                (None, Some(r)) => r,
                (Some(l), None) => l,
                (Some(l), Some(r)) if i - start > 2 => {
                    l + ((r - l) as f64 * (j - start + 1) as f64 / (i - start + 1) as f64) as u64
                }
                (Some(l), Some(r)) => {
                    if j - start < i - j {
                        l
                    } else {
                        r
                    }
                }
                _ => *item,
            };
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn mixed_words_match_official_chinese_character_contract() {
        assert_eq!(
            words("你好，Rust don't 3.14!"),
            vec!["你", "好", "Rust", "don't", "314"]
        );
    }
    #[test]
    fn timestamp_repair_preserves_good_spans_and_fixes_outliers() {
        assert_eq!(repair_timestamps(&[0, 80, 160, 240]), vec![0, 80, 160, 240]);
        assert_eq!(repair_timestamps(&[0, 900, 160, 240]), vec![0, 0, 160, 240]);
        for values in [
            vec![100, 40, 20],
            vec![400, 300, 200, 100, 500],
            vec![0, 900, 800, 700, 400, 500],
        ] {
            let fixed = repair_timestamps(&values);
            assert!(fixed.windows(2).all(|p| p[0] <= p[1]));
        }
    }
}
