use crate::{Error, Result};
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};

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
    /// Returns [`Error::Cancelled`] when the shared stop flag is set.
    pub fn check(&self) -> Result<()> {
        if self.is_cancelled() {
            Err(Error::Cancelled)
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
        assert!(matches!(worker.check(), Err(Error::Cancelled)));
        assert!(matches!(token.check(), Err(Error::Cancelled)));

        let flag = Arc::new(AtomicBool::new(false));
        CancellationToken::from_shared_flag(flag.clone()).cancel();
        assert!(flag.load(Ordering::Acquire));
        assert!(matches!(
            CancellationToken::from_shared_flag(flag).check(),
            Err(Error::Cancelled)
        ));
        assert!(CancellationToken::default().check().is_ok());
    }
}
