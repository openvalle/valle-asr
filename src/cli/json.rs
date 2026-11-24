use anyhow::{Result, ensure};
use std::io::Write;
use valle_asr::{Segment, TranscriptSummary};

/// Writes the Transcript JSON schema one segment at a time, so the segments of
/// a long file are never all in memory.
pub struct JsonTranscriptWriter<W: Write> {
    output: W,
    segments: u64,
}

impl<W: Write> JsonTranscriptWriter<W> {
    pub fn new(mut output: W) -> Result<Self> {
        output.write_all(b"{\"segments\":[\n")?;
        Ok(Self {
            output,
            segments: 0,
        })
    }

    pub fn write_segment(&mut self, segment: &Segment) -> Result<()> {
        if self.segments > 0 {
            self.output.write_all(b",\n")?;
        }
        serde_json::to_writer(&mut self.output, segment)?;
        self.segments += 1;
        Ok(())
    }

    /// Finish the document and flush it. Dropping without finishing leaves
    /// the destination JSON incomplete.
    pub fn finish(mut self, summary: &TranscriptSummary) -> Result<()> {
        ensure!(
            self.segments == summary.segment_count,
            "streamed segment count mismatch"
        );
        self.output.write_all(b"\n],\"model\":")?;
        serde_json::to_writer(&mut self.output, &summary.model)?;
        self.output.write_all(b",\"language\":")?;
        serde_json::to_writer(&mut self.output, &summary.language)?;
        write!(
            &mut self.output,
            ",\"duration_ms\":{},\"text\":",
            summary.duration_ms
        )?;
        serde_json::to_writer(&mut self.output, &summary.text)?;
        self.output.write_all(b"}\n")?;
        self.output.flush()?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use valle_asr::{Transcript, Word};

    #[test]
    fn streamed_json_reads_back_as_a_transcript() -> Result<()> {
        let segment = Segment {
            text: "quote\" newline\n反斜杠\\".into(),
            language: "English".into(),
            start_ms: 0,
            end_ms: 5,
            words: vec![Word {
                text: "quote\"".into(),
                start_ms: 1,
                end_ms: 2,
            }],
        };
        let mut output = Vec::new();
        let mut writer = JsonTranscriptWriter::new(&mut output)?;
        writer.write_segment(&segment)?;
        writer.write_segment(&segment)?;
        let summary = TranscriptSummary {
            model: "m".into(),
            text: "a b".into(),
            language: "English".into(),
            duration_ms: 5,
            segment_count: 2,
        };
        writer.finish(&summary)?;
        let parsed: Transcript = serde_json::from_slice(&output)?;
        assert_eq!(parsed.segments, [segment.clone(), segment]);
        assert_eq!((parsed.text.as_str(), parsed.duration_ms), ("a b", 5));
        let writer = JsonTranscriptWriter::new(Vec::new())?;
        assert!(writer.finish(&summary).is_err());
        Ok(())
    }
}
