//! `force_idr` names the next picture *pushed*, however far behind the
//! scheduler runs: under a macroblock-tree lookahead (H.264) or a cutree /
//! rate lookahead (H.265) the encoder holds the request until that picture
//! is offered, including when it is offered by `flush`.

use h26x::encode::h264::H264Encoder;
use h26x::encode::h265::H265Encoder;
use h26x::encode::{Access, Config, Preset, RateControl};

const W: u32 = 64;
const H: u32 = 64;
const FRAMES: u64 = 40;
/// Forced IDRs: one well inside the stream, one within a lookahead of its
/// end, so only `flush` offers it to the scheduler.
const FORCED: [u64; 2] = [5, 30];

fn frame(i: u64) -> Vec<u8> {
    let (w, h) = (W as usize, H as usize);
    let mut f = vec![128u8; w * h * 3 / 2];
    for y in 0..h {
        for x in 0..w {
            f[y * w + x] = ((x * 3 + y * 2 + i as usize * 5) & 0xff) as u8;
        }
    }
    f
}

fn config(tree: u32, lookahead: u32, rate: RateControl) -> Config {
    Config {
        width: W,
        height: H,
        gop: 250,
        bframes: 3,
        pyramid: true,
        tree_lookahead: tree,
        lookahead,
        rate,
        preset: Preset::Medium,
        threads: 1,
        ..Config::default()
    }
}

fn keyframes(
    mut push: impl FnMut(&[u8]) -> Vec<Access>,
    mut force: impl FnMut(),
    flush: impl FnOnce() -> Vec<Access>,
) -> Vec<u64> {
    let mut units = Vec::new();
    for i in 0..FRAMES {
        if FORCED.contains(&i) {
            force();
        }
        units.extend(push(&frame(i)));
    }
    units.extend(flush());
    assert_eq!(units.len() as u64, FRAMES, "one access unit per picture");
    let mut keys: Vec<u64> = units
        .iter()
        .filter(|a| a.keyframe)
        .map(|a| a.display)
        .collect();
    keys.sort_unstable();
    keys
}

#[test]
fn h264_forced_idr_lands_on_the_picture_pushed_under_a_tree_lookahead() {
    for tree in [0, 15, 40] {
        let e = std::cell::RefCell::new(
            H264Encoder::new(config(tree, 0, RateControl::ConstantQp(30))).unwrap(),
        );
        let keys = keyframes(
            |f| e.borrow_mut().push(f).unwrap(),
            || e.borrow_mut().force_idr(),
            || e.borrow_mut().flush().unwrap(),
        );
        assert_eq!(keys, vec![0, 5, 30], "tree {tree}");
    }
}

#[test]
fn h265_forced_idr_lands_on_the_picture_pushed_under_a_lookahead() {
    for (tree, lookahead, rate) in [
        (0, 0, RateControl::ConstantQp(30)),
        (16, 0, RateControl::ConstantQp(30)),
        (0, 12, RateControl::Bitrate { bps: 300_000 }),
    ] {
        let e = std::cell::RefCell::new(H265Encoder::new(config(tree, lookahead, rate)).unwrap());
        let keys = keyframes(
            |f| e.borrow_mut().push(f).unwrap(),
            || e.borrow_mut().force_idr(),
            || e.borrow_mut().flush().unwrap(),
        );
        assert_eq!(keys, vec![0, 5, 30], "tree {tree} lookahead {lookahead}");
    }
}
