//! Threads and a clock on targets whose standard library has neither.
//!
//! The decoders' worker pool starts its threads with `std::thread`, and the
//! HEVC decoder times two short waits with `std::time::Instant`. On
//! `wasm32-unknown-unknown` both are unsupported — `std::thread::spawn`
//! returns an error and `Instant::now()` panics — even in a build with
//! atomics and shared memory, where threads do exist: a wasm thread is
//! another instance of the module on the same memory, in a Web Worker, and
//! only the embedder (JavaScript) can start one.
//!
//! [`install`] is how the embedder supplies both. After it, every pool a
//! decoder creates asks [`Platform::spawn`] for its workers instead of
//! `std::thread`, and the waits read [`Platform::now_ms`] instead of
//! `Instant`. Without it nothing changes: native builds never call it and
//! behave exactly as before, and a single-threaded decoder (0 or 1 threads)
//! starts no pool and reads no clock on any target.
//!
//! A worker that [`Platform::spawn`] was asked for may start late (a Web
//! Worker loads asynchronously) or on a thread that has run other bodies
//! before. Neither matters to the pool: jobs wait in its queue until a
//! worker takes them, and a worker leaves the pool's loop — returning from
//! its body — when the pool is dropped, instead of being joined.

use std::sync::OnceLock;
use std::time::Duration;

/// What a thread runs: a pool worker's loop, which returns when its pool is
/// dropped.
pub type ThreadBody = Box<dyn FnOnce() + Send + 'static>;

/// The embedder's threads and clock. See the [module docs](self).
#[derive(Clone, Copy, Debug)]
pub struct Platform {
    /// Run `body` on a thread of its own, to completion. It must not wait
    /// for the body to start or to finish: the caller may go on to block
    /// until the body has done work. Should it be unable to start a thread,
    /// the body must still run some time (or the decoder waits for ever).
    pub spawn: fn(ThreadBody),
    /// Milliseconds on a monotonic clock (`performance.now()`), from any
    /// thread.
    pub now_ms: fn() -> f64,
}

static PLATFORM: OnceLock<Platform> = OnceLock::new();

/// Make the decoders start their threads and read the time through
/// `platform`, from now on, process-wide. Decoders created before keep the
/// threads they have. Returns false, changing nothing, if a platform was
/// installed already.
pub fn install(platform: Platform) -> bool {
    PLATFORM.set(platform).is_ok()
}

/// The installed platform, if any.
pub(crate) fn get() -> Option<&'static Platform> {
    PLATFORM.get()
}

/// A point in time: `Instant` natively, the embedder's clock once a
/// [`Platform`] is installed.
#[derive(Clone, Copy, Debug)]
pub(crate) enum Stamp {
    Std(std::time::Instant),
    Ms(f64),
}

impl Stamp {
    pub(crate) fn now() -> Stamp {
        match get() {
            Some(p) => Stamp::Ms((p.now_ms)()),
            None => Stamp::Std(std::time::Instant::now()),
        }
    }

    pub(crate) fn elapsed(&self) -> Duration {
        match *self {
            Stamp::Std(t) => t.elapsed(),
            Stamp::Ms(t) => {
                let now = get().map_or(t, |p| (p.now_ms)());
                Duration::from_secs_f64(((now - t) / 1000.0).max(0.0))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stamp_without_a_platform_is_instant() {
        // Native tests never install one: the clock is std's.
        assert!(get().is_none());
        let s = Stamp::now();
        assert!(matches!(s, Stamp::Std(_)));
        std::thread::sleep(Duration::from_millis(2));
        assert!(s.elapsed() >= Duration::from_millis(2));
    }

    #[test]
    fn a_millisecond_stamp_measures_from_its_reading() {
        // Without a platform the clock does not advance; elapsed is never
        // negative.
        assert_eq!(Stamp::Ms(5.0).elapsed(), Duration::ZERO);
    }
}
