use std::{
    error::Error,
    fmt,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
};

/// A cooperative cancellation returned instead of a successful partial result.
/// Use `anyhow::Error::is::<Cancelled>()` to recognize it through error context.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Cancelled;

impl fmt::Display for Cancelled {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("operation cancelled")
    }
}
impl Error for Cancelled {}

/// A cloneable, thread-safe stop signal shared by a request and its caller.
/// Cancellation is permanent for every clone. Use a new token for a new request.
#[derive(Debug, Clone, Default)]
pub struct CancellationToken(Arc<AtomicBool>);

impl CancellationToken {
    /// Share a host application's existing stop flag without a monitoring thread.
    /// The host must only change it from `false` to `true` during a request.
    /// An already-set flag cancels the token immediately.
    pub fn from_shared_flag(flag: Arc<AtomicBool>) -> Self {
        Self(flag)
    }

    /// Signal cancellation to all clones. This does not block or kill a thread.
    pub fn cancel(&self) {
        self.0.store(true, Ordering::Relaxed);
    }

    /// Whether cancellation has been requested.
    pub fn is_cancelled(&self) -> bool {
        self.0.load(Ordering::Relaxed)
    }

    /// Check a cooperative cancellation point.
    ///
    /// # Errors
    /// Returns [`Cancelled`] when the shared stop flag is set.
    pub fn check(&self) -> Result<(), Cancelled> {
        if self.is_cancelled() {
            Err(Cancelled)
        } else {
            Ok(())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shared_host_flag_and_clones_cancel_in_both_directions() {
        let flag = Arc::new(AtomicBool::new(false));
        let token = CancellationToken::from_shared_flag(flag.clone());
        let worker = token.clone();
        std::thread::spawn(move || flag.store(true, Ordering::Release))
            .join()
            .unwrap();
        assert_eq!(worker.check(), Err(Cancelled));
        assert_eq!(token.check(), Err(Cancelled));

        let flag = Arc::new(AtomicBool::new(false));
        CancellationToken::from_shared_flag(flag.clone()).cancel();
        assert!(flag.load(Ordering::Acquire));
        assert_eq!(
            CancellationToken::from_shared_flag(flag).check(),
            Err(Cancelled)
        );
        assert!(CancellationToken::default().check().is_ok());
    }
}
