use crate::{Segment, TranscriptSummary};
use anyhow::{Result, ensure};
use std::io::{BufRead, BufReader, BufWriter, Seek, SeekFrom, Write};
use tempfile::NamedTempFile;

/// Writes the existing Transcript JSON schema without retaining its segments or
/// full text in RAM. A temporary text spool is removed on success or error.
pub struct JsonTranscriptWriter<W: Write> {
    output: W,
    text: BufWriter<NamedTempFile>,
    segments: u64,
}

impl<W: Write> JsonTranscriptWriter<W> {
    /// Start a JSON document and create a temporary text spool.
    ///
    /// # Errors
    /// Returns an error if temporary storage or the destination writer fails.
    pub fn new(mut output: W) -> Result<Self> {
        let text = BufWriter::new(NamedTempFile::new()?);
        output.write_all(b"{\"segments\":[\n")?;
        Ok(Self {
            output,
            text,
            segments: 0,
        })
    }

    /// Append one segment and spool its text without retaining it in RAM.
    ///
    /// # Errors
    /// Propagates destination and temporary-storage write failures.
    pub fn write_segment(&mut self, segment: Segment) -> Result<()> {
        if self.segments > 0 {
            self.output.write_all(b",\n")?;
        }
        serde_json::to_writer(&mut self.output, &segment)?;
        if !segment.text.is_empty() {
            serde_json::to_writer(&mut self.text, &segment.text)?;
            self.text.write_all(b"\n")?;
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
        self.output.write_all(b"\n],\"model\":")?;
        serde_json::to_writer(&mut self.output, &summary.model)?;
        self.output.write_all(b",\"language\":")?;
        serde_json::to_writer(&mut self.output, &summary.language)?;
        write!(
            &mut self.output,
            ",\"duration_ms\":{},\"text\":\"",
            summary.duration_ms
        )?;
        self.text.flush()?;
        self.text.get_mut().as_file_mut().seek(SeekFrom::Start(0))?;
        let separator = if ["Chinese", "Cantonese", "Japanese"].contains(&summary.language.as_str())
        {
            b"".as_slice()
        } else {
            b" ".as_slice()
        };
        for (index, line) in BufReader::new(self.text.get_mut().as_file_mut())
            .lines()
            .enumerate()
        {
            let line = line?;
            ensure!(
                line.starts_with('"') && line.ends_with('"'),
                "invalid temporary text record"
            );
            if index > 0 {
                self.output.write_all(separator)?;
            }
            self.output.write_all(&line.as_bytes()[1..line.len() - 1])?;
        }
        self.output.write_all(b"\"}\n")?;
        self.output.flush()?;
        Ok(self.output)
    }
}
