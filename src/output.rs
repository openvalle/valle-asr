use crate::{Segment, TranscriptSummary};
use anyhow::{Result, ensure};
use std::io::Write;

/// Writes the existing Transcript JSON schema without retaining its segments in
/// RAM. Only the full text is kept until [`Self::finish`]: a few hundred bytes
/// per chunk, so well under a megabyte for hours of speech.
pub struct JsonTranscriptWriter<W: Write> {
    output: W,
    texts: Vec<String>,
    segments: u64,
}

impl<W: Write> JsonTranscriptWriter<W> {
    /// Start a JSON document.
    ///
    /// # Errors
    /// Returns an error if the destination writer fails.
    pub fn new(mut output: W) -> Result<Self> {
        output.write_all(b"{\"segments\":[\n")?;
        Ok(Self {
            output,
            texts: Vec::new(),
            segments: 0,
        })
    }

    /// Append one segment, keeping only its text.
    ///
    /// # Errors
    /// Propagates destination write failures.
    pub fn write_segment(&mut self, segment: Segment) -> Result<()> {
        if self.segments > 0 {
            self.output.write_all(b",\n")?;
        }
        serde_json::to_writer(&mut self.output, &segment)?;
        if !segment.text.is_empty() {
            self.texts.push(segment.text);
        }
        self.segments += 1;
        Ok(())
    }

    /// Finish the document, flush it, and return the destination writer.
    /// Dropping without finishing leaves the destination JSON incomplete.
    ///
    /// # Errors
    /// Rejects a summary with a different segment count and propagates I/O errors.
    pub fn finish(mut self, summary: &TranscriptSummary) -> Result<W> {
        ensure!(
            self.segments == summary.segment_count,
            "streamed segment count mismatch"
        );
        let separator = if ["Chinese", "Cantonese", "Japanese"].contains(&summary.language.as_str())
        {
            ""
        } else {
            " "
        };
        self.output.write_all(b"\n],\"model\":")?;
        serde_json::to_writer(&mut self.output, &summary.model)?;
        self.output.write_all(b",\"language\":")?;
        serde_json::to_writer(&mut self.output, &summary.language)?;
        write!(
            &mut self.output,
            ",\"duration_ms\":{},\"text\":",
            summary.duration_ms
        )?;
        serde_json::to_writer(&mut self.output, &self.texts.join(separator))?;
        self.output.write_all(b"}\n")?;
        self.output.flush()?;
        Ok(self.output)
    }
}
