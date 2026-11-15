//! The decoders on an embedder's threads and clock ([`h26x::platform`]).
//!
//! On wasm the embedder starts the workers (Web Workers) and supplies the
//! clock; nothing in this crate's native tests goes that way, because a
//! native build never installs a platform. This test does, in a process of
//! its own (an integration test is its own binary, and the platform is
//! process-wide): detached `std::thread`s stand in for Web Workers, a
//! counted `Instant` for `performance.now()`. The three vendored streams
//! must decode on four threads to the hashes `tests/decode.rs` holds them
//! to, every worker must have been asked of the platform, and every one
//! must leave its body when its decoder is dropped — the pool cannot join
//! a thread it did not start.

use std::sync::OnceLock;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use h26x::Picture;
use h26x::platform::{Platform, ThreadBody};

static SPAWNED: AtomicUsize = AtomicUsize::new(0);
static RUNNING: AtomicUsize = AtomicUsize::new(0);
static CLOCK_READS: AtomicUsize = AtomicUsize::new(0);

fn spawn(body: ThreadBody) {
    SPAWNED.fetch_add(1, Ordering::SeqCst);
    RUNNING.fetch_add(1, Ordering::SeqCst);
    std::thread::spawn(move || {
        body();
        RUNNING.fetch_sub(1, Ordering::SeqCst);
    });
}

fn now_ms() -> f64 {
    static START: OnceLock<Instant> = OnceLock::new();
    CLOCK_READS.fetch_add(1, Ordering::Relaxed);
    START.get_or_init(Instant::now).elapsed().as_secs_f64() * 1000.0
}

/// FNV-1a as in `tests/decode.rs`.
struct Hasher(u64);

impl Hasher {
    fn frame(&mut self, pic: Picture) {
        let (w, h) = (pic.width, pic.height);
        for b in w
            .to_le_bytes()
            .into_iter()
            .chain(h.to_le_bytes())
            .chain(pic.into_packed())
        {
            self.0 ^= b as u64;
            self.0 = self.0.wrapping_mul(0x100_0000_01b3);
        }
    }
}

macro_rules! decode {
    ($dec:expr, $data:expr) => {{
        let mut dec = $dec;
        let mut h = Hasher(0xcbf2_9ce4_8422_2325);
        let mut frames = 0;
        for nal in h26x::nal::annexb_nals($data) {
            dec.push_nal(nal).expect("nal");
            while let Some(p) = dec.try_next_picture() {
                h.frame(p);
                frames += 1;
            }
        }
        dec.flush().expect("flush");
        while let Some(p) = dec.next_picture() {
            h.frame(p);
            frames += 1;
        }
        (frames, h.0)
    }};
}

#[test]
fn decoders_run_on_the_installed_platform() {
    assert!(h26x::platform::install(Platform { spawn, now_ms }));
    // Installed once; a second call changes nothing.
    assert!(!h26x::platform::install(Platform { spawn, now_ms }));

    // The expectations of tests/decode.rs (computed from JM's and HM's output).
    for _ in 0..2 {
        let (frames, hash) = decode!(
            h26x::h264::H264Decoder::with_threads(4),
            include_bytes!("data/tiny_cabac.264")
        );
        assert_eq!((frames, hash), (12, 0xb088ec5117978d88), "tiny_cabac.264");
        let (frames, hash) = decode!(
            h26x::h264::H264Decoder::with_threads(4),
            include_bytes!("data/tiny_cavlc.264")
        );
        assert_eq!((frames, hash), (12, 0x83f3b2dd61096b23), "tiny_cavlc.264");
        let (frames, hash) = decode!(
            h26x::hevc::HevcDecoder::with_threads(4),
            include_bytes!("data/tiny.265")
        );
        assert_eq!((frames, hash), (12, 0xed1d0d0c4d5124ca), "tiny.265");
    }

    // Every worker came from the platform: four for each H.264 decoder, and
    // eight decoding plus one filtering for each HEVC one.
    assert_eq!(SPAWNED.load(Ordering::SeqCst), 2 * (4 + 4 + 8 + 1));
    // The decoders are gone, so their pools are: every worker returns from
    // its body (the pool does not join threads it did not start).
    let deadline = Instant::now() + Duration::from_secs(10);
    while RUNNING.load(Ordering::SeqCst) > 0 {
        assert!(
            Instant::now() < deadline,
            "pool workers outlived their decoders"
        );
        std::thread::sleep(Duration::from_millis(5));
    }
}

#[test]
fn a_single_threaded_decoder_needs_no_platform_threads() {
    // Runs alongside the test above, possibly before or after the install:
    // either way a decoder with one thread starts no pool (the count the
    // test above checks has no room for one) and reads no clock.
    let (frames, hash) = decode!(
        h26x::hevc::HevcDecoder::with_threads(1),
        include_bytes!("data/tiny.265")
    );
    assert_eq!((frames, hash), (12, 0xed1d0d0c4d5124ca));
    let (frames, hash) = decode!(
        h26x::h264::H264Decoder::with_threads(1),
        include_bytes!("data/tiny_cabac.264")
    );
    assert_eq!((frames, hash), (12, 0xb088ec5117978d88));
}
