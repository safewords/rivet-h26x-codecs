//! The CABAC arithmetic *encoder* (H.264 clause 9.3.4, H.265 clause 9.3.4) —
//! the inverse of [`crate::cabac::Cabac`].
//!
//! # Sharing with the decoder
//!
//! The probability model is not duplicated here. `rangeTabLps` and the two
//! transition tables live in [`crate::cabac`] and this module reads them, so
//! there is exactly one copy of those numbers in the crate and no way for the
//! two directions to drift apart.
//!
//! What it does *not* borrow is the decoder's fused `LPS` table, indexed
//! `(range & 0xc0) | pStateIdx`. That table caches, alongside each LPS range,
//! the range *already renormalised* and the shift that renormalised it —
//! which is what lets the decoder pick a branchless arm. The encoder cannot
//! use either field, because its renormalisation is not a shift: it emits a
//! bit per doubling, and which bit depends on `low` at that moment. So it
//! reads `LPS_RANGE` — the spec table the decoder's cache is itself derived
//! from. One source, two shapes, no second copy of the data.
//!
//! # Register model
//!
//! The standard's own: a 9-bit `range` and a `low` that the spec's `PutBit`
//! keeps inside ten bits by subtracting 512 or 256 as it emits. Carries are
//! handled the standard's way too, with an outstanding-bits counter rather
//! than by propagating into already-written bytes: a run of undecided bits is
//! counted, and settled by the first bit that resolves them.
//!
//! # The state must match the decoder's, bit for bit
//!
//! Both sides choose `ctxIdxInc` from *reconstructed* neighbours, so both
//! sides must hold the same context byte at the same bin. This module updates
//! state with the same tables and the same rule as
//! [`crate::cabac::Cabac::decision`], and the round-trip test checks the
//! whole context array afterwards rather than only the bins. A divergence
//! here does not produce a wrong pixel; it produces a decoder that is reading
//! a different bin than the one written, and it surfaces hundreds of
//! macroblocks later as a stream that falls apart with no obvious cause.
//!
//! Nothing in the crate calls this yet — the encoder that will is being built
//! alongside it — so the module allows dead code. Drop the allow when the
//! first slice writer lands.
#![allow(dead_code)]

use std::sync::LazyLock;

use crate::bitwriter::BitWriter;
use crate::cabac::{Ctx, LPS_RANGE, NEXT_STATE_LPS, NEXT_STATE_MPS};

/// `log2(range_before / range_after)` for every context-coded bin the
/// coder can meet, tabulated.
///
/// A bin's cost depends on three things only: the range it met (255
/// values, 256..=510), the context's probability state (64), and whether
/// it was the MPS or the LPS — the LPS sub-range is `LPS_RANGE[state]
/// [(range >> 6) & 3]`, a function of the first two, and the MPS gets the
/// rest. So the 32640 costs can be computed once, with the identical
/// expression [`CabacEncoder::encode_decision`] used to evaluate inline,
/// and looked up thereafter. The values are the same f64s bit for bit —
/// which is the point: `f64::log2` was 8–24% of every CABAC encode
/// profile (docs/encode_speed.md), charged per bin in the emitting
/// encoder too, and the table makes it one load without moving any
/// decision.
///
/// 255 KB, built lazily on first use. Indexed `((range - 256) * 64 +
/// state) * 2 + is_lps`.
static BIN_COST: LazyLock<Box<[f64]>> = LazyLock::new(|| {
    let mut t = vec![0f64; 255 * 64 * 2].into_boxed_slice();
    for range in 256u32..=510 {
        for p in 0..64usize {
            let lps = LPS_RANGE[p][((range >> 6) & 3) as usize] as u32;
            let i = ((range - 256) as usize * 64 + p) * 2;
            t[i] = (range as f64 / (range - lps) as f64).log2();
            t[i + 1] = (range as f64 / lps as f64).log2();
        }
    }
    t
});

/// What a context-coded bin costs, in bits, from its context's probability
/// state alone: `[state][is_lps]`. The state names a probability — the
/// standard's 64 states are the geometric ladder `p_LPS = 0.5 * a^state`,
/// `a = (0.01875 / 0.5)^(1/63)` (the design the state transition tables
/// approximate) — so a bin's information content is `-log2` of the
/// probability its state gives it. Independent of the coder's range, which
/// is what makes an estimate this cheap: one load, no renormalisation.
/// What a bin costs in bits at each probability state, `[MPS, LPS]`:
/// `-log2` of the state's probability, the states' probabilities being
/// `0.5 * a^s` with `a = (0.01875 / 0.5)^(1/63)` (9.3.2.2's model). A
/// literal rather than computed at first use, so a hot loop pricing bins
/// pays no initialisation check per bin; the test
/// `state_cost_table_is_the_formula` holds it to the formula bit for bit.
static STATE_COST: [[f32; 2]; 64] = [
    [f32::from_bits(0x3f800000), f32::from_bits(0x3f800000)],
    [f32::from_bits(0x3f6db480), f32::from_bits(0x3f899fd3)],
    [f32::from_bits(0x3f5d23b2), f32::from_bits(0x3f933fa6)],
    [f32::from_bits(0x3f4e12f0), f32::from_bits(0x3f9cdf79)],
    [f32::from_bits(0x3f405214), f32::from_bits(0x3fa67f4b)],
    [f32::from_bits(0x3f33b91b), f32::from_bits(0x3fb01f1e)],
    [f32::from_bits(0x3f282670), f32::from_bits(0x3fb9bef1)],
    [f32::from_bits(0x3f1d7d95), f32::from_bits(0x3fc35ec4)],
    [f32::from_bits(0x3f13a633), f32::from_bits(0x3fccfe97)],
    [f32::from_bits(0x3f0a8b4e), f32::from_bits(0x3fd69e6a)],
    [f32::from_bits(0x3f021ab5), f32::from_bits(0x3fe03e3c)],
    [f32::from_bits(0x3ef4890c), f32::from_bits(0x3fe9de0f)],
    [f32::from_bits(0x3ee5f59e), f32::from_bits(0x3ff37de2)],
    [f32::from_bits(0x3ed86286), f32::from_bits(0x3ffd1db5)],
    [f32::from_bits(0x3ecbb9f6), f32::from_bits(0x40035ec4)],
    [f32::from_bits(0x3ebfe880), f32::from_bits(0x40082ead)],
    [f32::from_bits(0x3eb4dcc8), f32::from_bits(0x400cfe97)],
    [f32::from_bits(0x3eaa8739), f32::from_bits(0x4011ce80)],
    [f32::from_bits(0x3ea0d9cd), f32::from_bits(0x40169e6a)],
    [f32::from_bits(0x3e97c7d7), f32::from_bits(0x401b6e53)],
    [f32::from_bits(0x3e8f45dc), f32::from_bits(0x40203e3c)],
    [f32::from_bits(0x3e874971), f32::from_bits(0x40250e26)],
    [f32::from_bits(0x3e7f9227), f32::from_bits(0x4029de0f)],
    [f32::from_bits(0x3e717834), f32::from_bits(0x402eadf9)],
    [f32::from_bits(0x3e64352a), f32::from_bits(0x40337de2)],
    [f32::from_bits(0x3e57ba7c), f32::from_bits(0x40384dcc)],
    [f32::from_bits(0x3e4bface), f32::from_bits(0x403d1db5)],
    [f32::from_bits(0x3e40e9d3), f32::from_bits(0x4041ed9e)],
    [f32::from_bits(0x3e367c33), f32::from_bits(0x4046bd88)],
    [f32::from_bits(0x3e2ca777), f32::from_bits(0x404b8d71)],
    [f32::from_bits(0x3e2361ee), f32::from_bits(0x40505d5b)],
    [f32::from_bits(0x3e1aa2a0), f32::from_bits(0x40552d44)],
    [f32::from_bits(0x3e12613d), f32::from_bits(0x4059fd2e)],
    [f32::from_bits(0x3e0a960a), f32::from_bits(0x405ecd17)],
    [f32::from_bits(0x3e0339da), f32::from_bits(0x40639d00)],
    [f32::from_bits(0x3df88bfd), f32::from_bits(0x40686cea)],
    [f32::from_bits(0x3deb6878), f32::from_bits(0x406d3cd3)],
    [f32::from_bits(0x3ddefd89), f32::from_bits(0x40720cbd)],
    [f32::from_bits(0x3dd34059), f32::from_bits(0x4076dca6)],
    [f32::from_bits(0x3dc826c6), f32::from_bits(0x407bac90)],
    [f32::from_bits(0x3dbda759), f32::from_bits(0x40803e3c)],
    [f32::from_bits(0x3db3b932), f32::from_bits(0x4082a631)],
    [f32::from_bits(0x3daa5406), f32::from_bits(0x40850e26)],
    [f32::from_bits(0x3da1700b), f32::from_bits(0x4087761b)],
    [f32::from_bits(0x3d9905f6), f32::from_bits(0x4089de0f)],
    [f32::from_bits(0x3d910eee), f32::from_bits(0x408c4604)],
    [f32::from_bits(0x3d898485), f32::from_bits(0x408eadf9)],
    [f32::from_bits(0x3d8260b0), f32::from_bits(0x409115ed)],
    [f32::from_bits(0x3d773b81), f32::from_bits(0x40937de2)],
    [f32::from_bits(0x3d6a6cc1), f32::from_bits(0x4095e5d7)],
    [f32::from_bits(0x3d5e4b11), f32::from_bits(0x40984dcc)],
    [f32::from_bits(0x3d52ccfc), f32::from_bits(0x409ab5c0)],
    [f32::from_bits(0x3d47e997), f32::from_bits(0x409d1db5)],
    [f32::from_bits(0x3d3d987e), f32::from_bits(0x409f85aa)],
    [f32::from_bits(0x3d33d1c6), f32::from_bits(0x40a1ed9e)],
    [f32::from_bits(0x3d2a8dfa), f32::from_bits(0x40a45593)],
    [f32::from_bits(0x3d21c60f), f32::from_bits(0x40a6bd88)],
    [f32::from_bits(0x3d197362), f32::from_bits(0x40a9257d)],
    [f32::from_bits(0x3d118faf), f32::from_bits(0x40ab8d71)],
    [f32::from_bits(0x3d0a150a), f32::from_bits(0x40adf566)],
    [f32::from_bits(0x3d02fdde), f32::from_bits(0x40b05d5b)],
    [f32::from_bits(0x3cf889c5), f32::from_bits(0x40b2c54f)],
    [f32::from_bits(0x3cebca38), f32::from_bits(0x40b52d44)],
    [f32::from_bits(0x3cdfb3a8), f32::from_bits(0x40b79539)],
];

/// A bin's cost and the context's next state, indexed by
/// `(ctx << 1) | bin` — [`STATE_COST`] and the transition tables fused,
/// so an estimating encoder prices a bin with two loads.
static EST_NEXT: [u8; 256] = {
    let mut t = [0u8; 256];
    let mut c = 0;
    while c < 128 {
        let p = c >> 1;
        let mps = (c & 1) as u8;
        let mut bin = 0;
        while bin < 2 {
            t[(c << 1) | bin] = if bin as u8 != mps {
                let flipped = if p == 0 { mps ^ 1 } else { mps };
                (NEXT_STATE_LPS[p] << 1) | flipped
            } else {
                (NEXT_STATE_MPS[p] << 1) | mps
            };
            bin += 1;
        }
        c += 1;
    }
    t
};

static EST_COST: [f32; 256] = {
    let mut t = [0f32; 256];
    let mut i = 0;
    while i < 256 {
        let c = i >> 1;
        let bin = (i & 1) as u8;
        t[i] = STATE_COST[c >> 1][(bin != (c & 1) as u8) as usize];
        i += 1;
    }
    t
};

#[inline]
pub(crate) fn state_bits(state: u8, bin: u32) -> f32 {
    STATE_COST[(state >> 1) as usize][(bin != (state & 1) as u32) as usize]
}

/// Where an encoder's output goes: a caller's writer, or nowhere.
///
/// Counting exists so a decision can ask what a shape *costs* without
/// producing it. The alternative — a table of what each syntax element
/// usually costs — is the thing this crate keeps deleting: a second model
/// of the writer, free to drift from the writer. Here there is one
/// arithmetic coder and one set of writers; counting changes only whether
/// the bits land in a buffer or in a tally, so a counted cost cannot
/// disagree with the emitted one about anything except where it was put.
enum Out<'a> {
    /// Write the bits.
    Bits(&'a mut BitWriter),
    /// Count them and throw them away.
    Count(u64),
    /// Run no arithmetic at all: advance the contexts and charge each bin
    /// its [`STATE_COST`] — see [`CabacEncoder::estimating`].
    Estimate,
}

impl Out<'_> {
    #[inline]
    fn bit(&mut self, v: u32) {
        match self {
            Out::Bits(w) => w.bit(v),
            Out::Count(n) => *n += 1,
            Out::Estimate => {}
        }
    }
    #[inline]
    fn bits(&mut self, n: u32, v: u32) {
        match self {
            Out::Bits(w) => w.bits(n, v),
            Out::Count(c) => *c += n as u64,
            Out::Estimate => {}
        }
    }
    #[inline]
    fn position(&self) -> u64 {
        match self {
            Out::Bits(w) => w.position(),
            Out::Count(n) => *n,
            Out::Estimate => 0,
        }
    }
}

/// The arithmetic encoder, writing into a caller's [`BitWriter`].
///
/// It borrows the writer rather than owning one because CABAC slice data
/// continues the same RBSP the slice header was written into, and the header
/// is plain bits.
///
/// A *counting* encoder ([`CabacEncoder::counting`]) runs the identical
/// arithmetic over the identical bins and tallies the bits instead of
/// writing them — see [`Out`].
pub struct CabacEncoder<'a> {
    w: Out<'a>,
    /// Fractional bits this encoder has consumed, accumulated from the
    /// coder's own interval arithmetic — see
    /// [`CabacEncoder::fractional_bits`].
    frac: f64,
    /// `codILow`.
    low: u32,
    /// `codIRange`.
    range: u32,
    /// `bitsOutstanding`: bits whose value is not yet decided, to be emitted
    /// as the complement of whichever bit settles them.
    outstanding: u32,
    /// `firstBitFlag`: the first `PutBit` produces no output, because the
    /// leading bit of the interval is implicit.
    first: bool,
}

impl<'a> CabacEncoder<'a> {
    /// Start encoding into `w`, which must be byte aligned (9.3.4.1). For
    /// H.264 that means after `cabac_alignment_one_bit`.
    pub fn new(w: &'a mut BitWriter) -> Self {
        debug_assert!(w.byte_aligned(), "CABAC data starts on a byte boundary");
        Self {
            w: Out::Bits(w),
            frac: 0.0,
            low: 0,
            range: 510,
            outstanding: 0,
            first: true,
        }
    }

    /// An encoder that counts the bits it would write and produces none.
    ///
    /// The register state is the standard's start-of-codeword one, the same
    /// [`CabacEncoder::new`] begins from, so counting a whole slice from a
    /// freshly initialised context array yields *exactly* the bit count
    /// writing it would produce — that equality is asserted in this
    /// module's tests, and it is what makes a counted cost a measurement
    /// rather than an estimate.
    ///
    /// Counting a *fragment* mid-stream is a different question, and an
    /// honest answer to it is smaller than exact: an arithmetic coder
    /// carries fractional state between bins, so no prefix of a codeword
    /// has a bit count of its own. A fragment counted from this neutral
    /// start is therefore within a bit or two of the bits it would add in
    /// place. That is the right tool for *comparing two candidates at the
    /// same point* — both are counted from the same start, so the
    /// difference between them is what the comparison needs and the shared
    /// offset cancels. Callers wanting an absolute figure should count the
    /// whole slice.
    pub fn counting() -> CabacEncoder<'static> {
        CabacEncoder {
            w: Out::Count(0),
            frac: 0.0,
            low: 0,
            range: 510,
            outstanding: 0,
            first: true,
        }
    }

    /// An encoder that only *prices* what it is given: each context-coded
    /// bin advances its context exactly as encoding would and is charged
    /// the cost its probability state gives it ([`STATE_COST`]), each
    /// bypass bin one bit, and no interval arithmetic runs at all.
    /// [`CabacEncoder::fractional_bits`] is the estimate.
    ///
    /// Not the measurement [`CabacEncoder::counting`] is — the real coder's
    /// cost of a bin also depends on the interval it meets — but the same
    /// within a small fraction of a bit per bin, and several times cheaper:
    /// what a mode decision pricing many candidates per macroblock wants.
    pub fn estimating() -> CabacEncoder<'static> {
        CabacEncoder {
            w: Out::Estimate,
            frac: 0.0,
            low: 0,
            range: 510,
            outstanding: 0,
            first: true,
        }
    }

    /// `PutBit(b)` (9.3.4.3): emit `b`, then settle every outstanding bit as
    /// its complement.
    #[inline]
    fn put_bit(&mut self, b: u32) {
        if self.first {
            self.first = false;
        } else {
            self.w.bit(b);
        }
        while self.outstanding > 0 {
            self.w.bit(b ^ 1);
            self.outstanding -= 1;
        }
    }

    /// `RenormE` (9.3.4.3): double the range until it is back in
    /// `[256, 510]`, emitting or deferring one bit each time.
    #[inline]
    fn renorm(&mut self) {
        while self.range < 256 {
            if self.low < 256 {
                self.put_bit(0);
            } else if self.low >= 512 {
                self.low -= 512;
                self.put_bit(1);
            } else {
                // Straddling the midpoint: the bit is not decided yet.
                self.low -= 256;
                self.outstanding += 1;
            }
            self.range <<= 1;
            self.low <<= 1;
        }
        debug_assert!(self.low < 1024, "codILow left its ten bits");
    }

    /// Encode one context-coded bin (9.3.4.2), advancing `ctx` exactly as
    /// [`crate::cabac::Cabac::decision`] advances it on the way back.
    pub fn encode_decision(&mut self, ctx: &mut Ctx, bin: u32) {
        debug_assert!(bin <= 1);
        let state = *ctx;
        if let Out::Estimate = self.w {
            let i = ((state as usize) << 1 | bin as usize) & 255;
            self.frac += EST_COST[i] as f64;
            *ctx = EST_NEXT[i];
            return;
        }
        let p = (state >> 1) as usize;
        let mps = (state & 1) as u32;
        let lps = LPS_RANGE[p][((self.range >> 6) & 3) as usize] as u32;
        let before = self.range;
        self.range -= lps;
        if bin != mps {
            // The LPS sub-interval sits above the MPS one, so the low end
            // moves up by the whole MPS range.
            self.low += self.range;
            self.range = lps;
            let flipped = if p == 0 { mps ^ 1 } else { mps };
            *ctx = (NEXT_STATE_LPS[p] << 1) | flipped as u8;
        } else {
            *ctx = (NEXT_STATE_MPS[p] << 1) | mps as u8;
        }
        // The bin's own information content: the interval it survived
        // divided by the interval it left. Renormalisation only rescales
        // both, so charging it here and not again below is exact. Only a
        // counting encoder ever reads it, and the table holds exactly the
        // value the expression `(before as f64 / self.range as f64).log2()`
        // would produce here — see `BIN_COST`.
        if let Out::Count(_) = self.w {
            self.frac += BIN_COST[((before - 256) as usize * 64 + p) * 2 + (bin != mps) as usize];
        }
        self.renorm();
    }

    /// Encode one bypass bin (9.3.4.4). No context, no renormalisation: the
    /// range is untouched and `low` takes the doubling instead.
    pub fn encode_bypass(&mut self, bin: u32) {
        debug_assert!(bin <= 1);
        // A bypass bin halves the interval by construction: one bit, exactly.
        self.frac += 1.0;
        if let Out::Estimate = self.w {
            return;
        }
        self.low <<= 1;
        if bin != 0 {
            self.low += self.range;
        }
        if self.low >= 1024 {
            self.put_bit(1);
            self.low -= 1024;
        } else if self.low < 512 {
            self.put_bit(0);
        } else {
            self.low -= 512;
            self.outstanding += 1;
        }
    }

    /// Encode `n` bypass bins (`n <= 32`) of `v`, most significant first —
    /// the inverse of [`crate::cabac::Cabac::bypass_bits`].
    pub fn encode_bypass_bits(&mut self, n: u32, v: u32) {
        debug_assert!(n <= 32);
        for i in (0..n).rev() {
            self.encode_bypass((v >> i) & 1);
        }
    }

    /// Encode a terminate bin (9.3.4.5). A `1` ends the arithmetic codeword
    /// and flushes; nothing may be encoded afterwards.
    pub fn encode_terminate(&mut self, bin: u32) {
        debug_assert!(bin <= 1);
        if let Out::Estimate = self.w {
            // A terminate bin of 0 is nearly free (2 / range of the
            // interval goes); 1 ends the codeword, about seven bits.
            self.frac += if bin != 0 { 7.0 } else { 0.01 };
            return;
        }
        let before = self.range;
        self.range -= 2;
        self.frac += if bin != 0 {
            // Ending the codeword costs what the 2/range sub-interval is worth.
            (before as f64 / 2.0).log2()
        } else {
            (before as f64 / self.range as f64).log2()
        };
        if bin != 0 {
            self.low += self.range;
            self.flush();
        } else {
            self.renorm();
        }
    }

    /// `EncodeFlush` (9.3.4.6): close the interval and leave the writer at
    /// the bit position the standard hands back to plain bit reading.
    ///
    /// The final two bits are `((low >> 7) & 3) | 1`; the low bit forced to
    /// one is the `rbsp_stop_one_bit`'s counterpart inside the codeword —
    /// it is what makes the decoder's terminate read back as 1.
    fn flush(&mut self) {
        self.range = 2;
        self.renorm();
        self.put_bit((self.low >> 9) & 1);
        self.w.bits(2, ((self.low >> 7) & 3) | 1);
    }

    /// Bits written into the underlying writer so far — or, for a counting
    /// encoder, the bits it would have written.
    pub fn position(&self) -> u64 {
        self.w.position()
    }

    /// Bits this encoder has accounted for. Identical to
    /// [`CabacEncoder::position`] for a counting encoder, which starts at
    /// zero; on a writing encoder the writer may already hold a slice
    /// header, so prefer taking a difference of `position`.
    pub fn bits_counted(&self) -> u64 {
        self.w.position()
    }

    /// What the bins encoded so far actually cost, in fractional bits.
    ///
    /// **This is the figure a decision should compare candidates on**, and
    /// [`CabacEncoder::bits_counted`] is not — a fact that cost a wrong
    /// answer before it was written down here. An arithmetic coder emits
    /// no output at all until its interval has narrowed past a byte, so a
    /// short fragment counted by emitted bits reads as ZERO. Pricing intra
    /// mode syntax that way made every most-probable-mode shape cost 0
    /// bits, deleted the rate term from the decision, and made the encoder
    /// measurably worse while looking like a more accurate model.
    ///
    /// The accounting is the coder's own, not a probability table: every
    /// context-coded bin is charged `log2(range_before / range_after)` —
    /// the interval it survived over the interval it left, which IS its
    /// information content — and every bypass bin exactly one bit.
    /// Renormalisation rescales both sides of that ratio, so it neither
    /// adds nor hides cost. Nothing here models what a bin "usually"
    /// costs; it reads what this bin cost in this state.
    ///
    /// Summed over a whole codeword it agrees with the emitted bit count
    /// to within the flush's few bits, which the round trip asserts.
    pub fn fractional_bits(&self) -> f64 {
        self.frac
    }
}

/// [`CabacEncoder::new`]'s bits by a faster route, for the slice writers
/// that only ever finish a slice: rather than the standard's
/// bit-at-a-time `PutBit` with its outstanding-bits counter, `low` keeps
/// the bits that leave its ten-bit window above it and hands them to the
/// writer a byte at a time, a carry out of the window incrementing the one
/// byte that can still take it (and turning the `0xff` bytes after it to
/// `0x00`) — the same codeword, since the outstanding bits are exactly the
/// run a carry would flip. The module's round trip holds the two to the
/// same bytes.
pub(crate) struct CabacWriter<'a> {
    w: &'a mut BitWriter,
    /// The registers every bin touches, kept apart so a hot loop can hold
    /// them by value ([`BinEnc::split`]).
    core: WCore,
    /// The byte that a carry may still increment (`-1`: none yet), ...
    pend: i32,
    /// ... and the `0xff` bytes after it, which a carry turns to `0x00`.
    ffs: u32,
}

/// [`CabacWriter`]'s per-bin state.
#[derive(Clone, Copy)]
pub(crate) struct WCore {
    /// `codILow`, with [`WCore::queued`] bits above its window.
    low: u32,
    /// `codIRange`.
    range: u32,
    /// How many bits `low` holds above its ten-bit window ...
    queued: u32,
    /// ... and how many make a byte to hand on: nine for the first, whose
    /// leading bit is the implicit one `firstBitFlag` drops, eight after.
    need: u32,
}

impl<'a> CabacWriter<'a> {
    /// Start writing into `w`, byte aligned as for [`CabacEncoder::new`].
    pub(crate) fn new(w: &'a mut BitWriter) -> Self {
        debug_assert!(w.byte_aligned(), "CABAC data starts on a byte boundary");
        Self {
            w,
            core: WCore {
                low: 0,
                range: 510,
                queued: 0,
                need: 9,
            },
            pend: -1,
            ffs: 0,
        }
    }

    /// [`WCore::settle`]'s byte, out of line: one bin in several gets
    /// here. The core goes in and comes back by value, so the caller's
    /// copy never has to live in memory.
    #[inline(never)]
    fn fast_out(&mut self, mut c: WCore) -> WCore {
        let need = c.need;
        let shift = 10 + c.queued - need;
        let v = c.low >> shift;
        c.low &= (1 << shift) - 1;
        c.queued -= need;
        let (carry, byte) = if need == 9 {
            c.need = 8;
            debug_assert!(v < 0x100, "a carry past the codeword's first bit");
            (0, v & 0xff)
        } else {
            (v >> 8, v & 0xff)
        };
        self.fast_byte(carry, byte);
        c
    }

    #[inline]
    fn fast_byte(&mut self, carry: u32, byte: u32) {
        let w = &mut *self.w;
        if carry != 0 {
            debug_assert!(self.pend >= 0, "a carry with no byte to take it");
            w.bits(8, (self.pend as u32 + 1) & 0xff);
            for _ in 0..self.ffs {
                w.bits(8, 0);
            }
            self.pend = -1;
            self.ffs = 0;
        }
        if byte == 0xff {
            self.ffs += 1;
        } else {
            if self.pend >= 0 {
                w.bits(8, self.pend as u32);
            }
            for _ in 0..self.ffs {
                w.bits(8, 0xff);
            }
            self.pend = byte as i32;
            self.ffs = 0;
        }
    }

    /// [`CabacWriter`]'s `EncodeFlush`: the window's top three bits, the
    /// last forced to one, join the queue, and everything left goes out.
    fn fast_flush(&mut self) {
        let mut c = self.core;
        c.range = 2;
        c.renorm(self);
        c.low |= 0x80;
        c.low <<= 3;
        c.queued += 3;
        c.settle(self);
        c.settle(self);
        // Nothing can carry any more: the pending byte and its run go out
        // as they stand, then the bits short of a byte.
        let mut rest = c.queued;
        let mut v = c.low >> 10;
        if c.need == 9 {
            // Never reached a byte: drop the leading bit here.
            debug_assert!(rest >= 1);
            rest -= 1;
            v &= (1 << rest) - 1;
            c.need = 8;
        }
        // A carry out of the last bits still reaches the pending byte.
        self.fast_byte(v >> rest, 0);
        // `fast_byte` queued a zero byte as pending; drop it.
        self.pend = -1;
        let w = &mut *self.w;
        if rest > 0 {
            w.bits(rest, v & ((1 << rest) - 1));
        }
        c.queued = 0;
        c.low = 0;
        self.core = c;
    }
}

impl WCore {
    /// Move a whole byte from above the window to the writer, when there
    /// is one.
    #[inline(always)]
    fn settle(&mut self, out: &mut CabacWriter<'_>) {
        if self.queued >= self.need {
            *self = out.fast_out(*self);
        }
    }

    /// The renormalisation: one shift, then whole bytes out.
    #[inline(always)]
    fn renorm(&mut self, out: &mut CabacWriter<'_>) {
        if self.range < 256 {
            let s = self.range.leading_zeros() - 23;
            self.range <<= s;
            self.low <<= s;
            self.queued += s;
            self.settle(out);
        }
    }
}

impl<'a> BinCore<CabacWriter<'a>> for WCore {
    #[inline(always)]
    fn decision(&mut self, out: &mut CabacWriter<'a>, ctx: &mut Ctx, bin: u32) {
        let state = *ctx;
        let p = (state >> 1) as usize;
        let mps = (state & 1) as u32;
        let lps = LPS_RANGE[p][((self.range >> 6) & 3) as usize] as u32;
        // Whether a bin is the LPS is the data's business, so the update
        // and the renormalisation are selects and an unconditional shift
        // (by zero when the range is still in its upper half: a nine-bit
        // range has 23 leading zeros) rather than branches.
        let rmps = self.range - lps;
        let is_lps = bin != mps;
        self.low += if is_lps { rmps } else { 0 };
        let range = if is_lps { lps } else { rmps };
        *ctx = EST_NEXT[((state as usize) << 1 | bin as usize) & 255];
        let s = range.leading_zeros() - 23;
        self.range = range << s;
        self.low <<= s;
        self.queued += s;
        self.settle(out);
    }

    #[inline(always)]
    fn bypass(&mut self, out: &mut CabacWriter<'a>, bin: u32) {
        self.low = (self.low << 1) + (self.range & 0u32.wrapping_sub((bin != 0) as u32));
        self.queued += 1;
        self.settle(out);
    }
}

impl<'a> BinEnc for CabacWriter<'a> {
    type Core = WCore;
    type Out = CabacWriter<'a>;

    #[inline(always)]
    fn split(&mut self) -> (WCore, &mut Self) {
        (self.core, self)
    }

    #[inline(always)]
    fn join(&mut self, core: WCore) {
        self.core = core;
    }

    #[inline(always)]
    fn encode_decision(&mut self, ctx: &mut Ctx, bin: u32) {
        let mut c = self.core;
        c.decision(self, ctx, bin);
        self.core = c;
    }

    #[inline(always)]
    fn encode_bypass(&mut self, bin: u32) {
        let mut c = self.core;
        c.bypass(self, bin);
        self.core = c;
    }

    fn encode_terminate(&mut self, bin: u32) {
        self.core.range -= 2;
        if bin != 0 {
            self.core.low += self.core.range;
            self.fast_flush();
        } else {
            let mut c = self.core;
            c.renorm(self);
            self.core = c;
        }
    }
}

/// A bin coder's per-bin state taken out of it by value
/// ([`BinEnc::split`]), so a loop that codes many bins — a residual
/// block — keeps it in registers; `out` is the coder itself, for what a
/// bin rarely needs (a byte to write).
pub(crate) trait BinCore<O: ?Sized>: Copy {
    /// A context-coded bin.
    fn decision(&mut self, out: &mut O, ctx: &mut Ctx, bin: u32);
    /// A bypass bin.
    fn bypass(&mut self, out: &mut O, bin: u32);
}

/// The bin-level interface the macroblock writers code through, so one
/// writer serves both an encoder that writes ([`CabacEncoder`]) and one
/// that only prices ([`CabacEstimator`]) — each compiled for its own, the
/// pricing one with nothing in a bin but two table loads.
pub(crate) trait BinEnc {
    /// The per-bin state [`BinEnc::split`] hands out.
    type Core: BinCore<Self::Out>;
    /// What the state codes into.
    type Out: ?Sized;
    /// Take the per-bin state out by value, with what it codes into.
    /// Bins coded through the pair must be followed by [`BinEnc::join`]
    /// before the coder is used directly again.
    fn split(&mut self) -> (Self::Core, &mut Self::Out);
    /// Put the state [`BinEnc::split`] took back.
    fn join(&mut self, core: Self::Core);
    /// A context-coded bin.
    fn encode_decision(&mut self, ctx: &mut Ctx, bin: u32);
    /// A bypass bin.
    fn encode_bypass(&mut self, bin: u32);
    /// `n` bypass bins of `v`, most significant first.
    fn encode_bypass_bits(&mut self, n: u32, v: u32) {
        debug_assert!(n <= 32);
        for i in (0..n).rev() {
            self.encode_bypass((v >> i) & 1);
        }
    }
    /// A terminate bin.
    fn encode_terminate(&mut self, bin: u32);
}

/// [`CabacEncoder`]'s [`BinCore`]: no state of its own, every bin through
/// the encoder.
#[derive(Clone, Copy)]
pub(crate) struct Forward;

impl<'a> BinCore<CabacEncoder<'a>> for Forward {
    #[inline(always)]
    fn decision(&mut self, out: &mut CabacEncoder<'a>, ctx: &mut Ctx, bin: u32) {
        out.encode_decision(ctx, bin)
    }
    #[inline(always)]
    fn bypass(&mut self, out: &mut CabacEncoder<'a>, bin: u32) {
        out.encode_bypass(bin)
    }
}

impl<'a> BinEnc for CabacEncoder<'a> {
    type Core = Forward;
    type Out = CabacEncoder<'a>;
    #[inline(always)]
    fn split(&mut self) -> (Forward, &mut Self) {
        (Forward, self)
    }
    #[inline(always)]
    fn join(&mut self, _: Forward) {}
    #[inline]
    fn encode_decision(&mut self, ctx: &mut Ctx, bin: u32) {
        CabacEncoder::encode_decision(self, ctx, bin)
    }
    #[inline]
    fn encode_bypass(&mut self, bin: u32) {
        CabacEncoder::encode_bypass(self, bin)
    }
    #[inline]
    fn encode_bypass_bits(&mut self, n: u32, v: u32) {
        CabacEncoder::encode_bypass_bits(self, n, v)
    }
    #[inline]
    fn encode_terminate(&mut self, bin: u32) {
        CabacEncoder::encode_terminate(self, bin)
    }
}

/// [`CabacEncoder::estimating`] as a type of its own: the same prices and
/// the same context transitions, with no output to dispatch on.
#[derive(Default)]
pub(crate) struct CabacEstimator {
    frac: f64,
}

impl CabacEstimator {
    /// A fresh estimate.
    pub(crate) fn new() -> Self {
        Self::default()
    }

    /// The bins' estimated cost so far, in bits.
    pub(crate) fn fractional_bits(&self) -> f64 {
        self.frac
    }
}

/// [`CabacEstimator`]'s [`BinCore`]: the running total.
#[derive(Clone, Copy)]
pub(crate) struct EstCore(f64);

impl BinCore<CabacEstimator> for EstCore {
    #[inline(always)]
    fn decision(&mut self, _: &mut CabacEstimator, ctx: &mut Ctx, bin: u32) {
        let i = ((*ctx as usize) << 1 | bin as usize) & 255;
        self.0 += EST_COST[i] as f64;
        *ctx = EST_NEXT[i];
    }
    #[inline(always)]
    fn bypass(&mut self, _: &mut CabacEstimator, _bin: u32) {
        self.0 += 1.0;
    }
}

impl BinEnc for CabacEstimator {
    type Core = EstCore;
    type Out = CabacEstimator;
    #[inline(always)]
    fn split(&mut self) -> (EstCore, &mut Self) {
        (EstCore(self.frac), self)
    }
    #[inline(always)]
    fn join(&mut self, core: EstCore) {
        self.frac = core.0;
    }
    #[inline(always)]
    fn encode_decision(&mut self, ctx: &mut Ctx, bin: u32) {
        let i = ((*ctx as usize) << 1 | bin as usize) & 255;
        self.frac += EST_COST[i] as f64;
        *ctx = EST_NEXT[i];
    }
    #[inline(always)]
    fn encode_bypass(&mut self, _bin: u32) {
        self.frac += 1.0;
    }
    #[inline(always)]
    fn encode_bypass_bits(&mut self, n: u32, _v: u32) {
        for _ in 0..n {
            self.frac += 1.0;
        }
    }
    #[inline(always)]
    fn encode_terminate(&mut self, bin: u32) {
        self.frac += if bin != 0 { 7.0 } else { 0.01 };
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cabac::{Cabac, init_ctx_h264};

    /// The table IS the expression: every entry against the inline
    /// computation it replaced, as f64 bits, so the encoder's decisions
    /// cannot have moved by a rounding.
    /// The literal [`STATE_COST`] is the model's formula, to the bit, and
    /// the fused estimator tables agree with the coder's own transitions.
    #[test]
    fn state_cost_table_is_the_formula() {
        let a = (0.01875f64 / 0.5).powf(1.0 / 63.0);
        for (s, e) in STATE_COST.iter().enumerate() {
            let p = 0.5 * a.powi(s as i32);
            assert_eq!(
                e[0].to_bits(),
                ((-(1.0 - p).log2()) as f32).to_bits(),
                "state {s}"
            );
            assert_eq!(e[1].to_bits(), ((-p.log2()) as f32).to_bits(), "state {s}");
        }
        for c in 0..128u8 {
            for bin in 0..2u32 {
                let mut x = c;
                let mut e = CabacEncoder::counting();
                e.encode_decision(&mut x, bin);
                let i = ((c as usize) << 1) | bin as usize;
                assert_eq!(EST_NEXT[i], x, "ctx {c} bin {bin}");
                assert_eq!(EST_COST[i].to_bits(), state_bits(c, bin).to_bits());
            }
        }
    }

    #[test]
    fn bin_cost_table_is_the_inline_expression() {
        for range in 256u32..=510 {
            for p in 0..64usize {
                let lps = LPS_RANGE[p][((range >> 6) & 3) as usize] as u32;
                let i = ((range - 256) as usize * 64 + p) * 2;
                let mps_cost = (range as f64 / (range - lps) as f64).log2();
                let lps_cost = (range as f64 / lps as f64).log2();
                assert_eq!(
                    BIN_COST[i].to_bits(),
                    mps_cost.to_bits(),
                    "range {range} state {p} mps"
                );
                assert_eq!(
                    BIN_COST[i + 1].to_bits(),
                    lps_cost.to_bits(),
                    "range {range} state {p} lps"
                );
            }
        }
    }

    /// What a bin is, for the round trip.
    #[derive(Clone, Copy, Debug, PartialEq)]
    enum Op {
        /// Context-coded, against context `n` of the pool.
        Decision(usize, u32),
        Bypass(u32),
        BypassBits(u32, u32),
        /// `terminate` with a zero bin — the codeword continues.
        Terminate,
    }

    const POOL: usize = 24;

    fn contexts() -> [Ctx; POOL] {
        // A spread of initial states, from the real initialisation formula so
        // the states are ones that actually occur.
        let mut c = [0u8; POOL];
        for (i, v) in c.iter_mut().enumerate() {
            *v = init_ctx_h264(20 - (i as i32 % 40), (i as i32 * 7) % 90 - 45, 26);
        }
        c
    }

    /// Encode a sequence, decode it back, and require both the bins and the
    /// final context states to match. This is the whole correctness argument
    /// for the encoder: it is exact, it has no floor, and it does not depend
    /// on the machine.
    fn round_trip(ops: &[Op]) {
        let mut enc_ctx = contexts();
        let mut w = BitWriter::new();
        {
            let mut e = CabacEncoder::new(&mut w);
            for op in ops {
                match *op {
                    Op::Decision(i, b) => e.encode_decision(&mut enc_ctx[i], b),
                    Op::Bypass(b) => e.encode_bypass(b),
                    Op::BypassBits(n, v) => e.encode_bypass_bits(n, v),
                    Op::Terminate => e.encode_terminate(0),
                }
            }
            // A terminate of 1 both ends the codeword and flushes it.
            e.encode_terminate(1);
        }
        let written = w.position();
        w.align_zero();
        let data = w.into_rbsp();

        // The byte-at-a-time writer must produce the very same codeword,
        // carries and all, behind a header that leaves the writer at an
        // arbitrary byte.
        for lead in [0u32, 3] {
            let mut want = BitWriter::new();
            let mut got = BitWriter::new();
            for b in [&mut want, &mut got] {
                b.bits(8 * lead, 0x5a5a5a & ((1u32 << (8 * lead)) - 1));
            }
            let mut ca = contexts();
            let mut cb = contexts();
            {
                let mut a = CabacEncoder::new(&mut want);
                let mut f = CabacWriter::new(&mut got);
                for op in ops {
                    match *op {
                        Op::Decision(i, b) => {
                            a.encode_decision(&mut ca[i], b);
                            BinEnc::encode_decision(&mut f, &mut cb[i], b);
                        }
                        Op::Bypass(b) => {
                            a.encode_bypass(b);
                            BinEnc::encode_bypass(&mut f, b);
                        }
                        Op::BypassBits(n, v) => {
                            a.encode_bypass_bits(n, v);
                            BinEnc::encode_bypass_bits(&mut f, n, v);
                        }
                        Op::Terminate => {
                            a.encode_terminate(0);
                            BinEnc::encode_terminate(&mut f, 0);
                        }
                    }
                }
                a.encode_terminate(1);
                BinEnc::encode_terminate(&mut f, 1);
            }
            assert_eq!(want.position(), got.position(), "fast writer's length");
            want.align_zero();
            got.align_zero();
            assert_eq!(want.into_rbsp(), got.into_rbsp(), "fast writer's bytes");
        }

        // The counting encoder must account for EXACTLY the bits the
        // writing one produced, over the same bins from the same start.
        // This is the property that lets a decision price a shape by
        // running the real writers instead of guessing: if the two ever
        // disagreed, every cost derived from counting would be wrong by an
        // amount nothing else in the crate could detect.
        let mut cnt_ctx = contexts();
        let mut c = CabacEncoder::counting();
        for op in ops {
            match *op {
                Op::Decision(i, b) => c.encode_decision(&mut cnt_ctx[i], b),
                Op::Bypass(b) => c.encode_bypass(b),
                Op::BypassBits(n, v) => c.encode_bypass_bits(n, v),
                Op::Terminate => c.encode_terminate(0),
            }
        }
        c.encode_terminate(1);
        assert_eq!(
            c.bits_counted(),
            written,
            "counted bits differ from the bits written"
        );
        // And the fractional accounting agrees with the emitted count to
        // within the flush. It is the fractional figure a decision must
        // compare on: emitted bits read as ZERO for a fragment shorter
        // than the coder's first output byte, which silently deletes the
        // rate term from a decision rather than making it inaccurate.
        let frac = c.fractional_bits();
        assert!(
            (frac - written as f64).abs() <= 8.0,
            "fractional cost {frac:.2} and emitted bits {written} disagree by more than the flush"
        );
        assert_eq!(
            cnt_ctx, enc_ctx,
            "counting advanced the contexts differently from writing"
        );

        let mut dec_ctx = contexts();
        let mut d = Cabac::new(&data);
        for (k, op) in ops.iter().enumerate() {
            match *op {
                Op::Decision(i, b) => {
                    assert_eq!(d.decision(&mut dec_ctx[i]), b, "op {k}: {op:?}");
                }
                Op::Bypass(b) => assert_eq!(d.bypass(), b, "op {k}: {op:?}"),
                Op::BypassBits(n, v) => assert_eq!(d.bypass_bits(n), v, "op {k}: {op:?}"),
                Op::Terminate => assert_eq!(d.terminate(), 0, "op {k}: {op:?}"),
            }
        }
        assert_eq!(
            d.terminate(),
            1,
            "the closing terminate did not read back as 1"
        );
        assert!(!d.overrun(), "decoder ran past what the encoder wrote");
        assert_eq!(
            enc_ctx, dec_ctx,
            "context states diverged: the two sides would desync on the next bin"
        );
    }

    #[test]
    fn round_trips_a_handful_of_bins() {
        round_trip(&[Op::Decision(0, 0)]);
        round_trip(&[Op::Decision(0, 1)]);
        round_trip(&[Op::Bypass(0)]);
        round_trip(&[Op::Bypass(1)]);
        round_trip(&[Op::Terminate, Op::Decision(3, 1), Op::Terminate]);
    }

    /// The carry path: long runs of one symbol drive `bitsOutstanding` up,
    /// and a run of bypass ones straddling the midpoint is what makes it
    /// count past one.
    #[test]
    fn round_trips_long_runs() {
        for bin in 0..2 {
            round_trip(&(0..500).map(|_| Op::Decision(0, bin)).collect::<Vec<_>>());
            round_trip(&(0..500).map(|_| Op::Bypass(bin)).collect::<Vec<_>>());
        }
        // Alternating, which keeps the state machine moving rather than
        // saturating at one end.
        round_trip(
            &(0..500)
                .map(|i| Op::Decision(i % POOL, (i % 2) as u32))
                .collect::<Vec<_>>(),
        );
    }

    #[test]
    fn round_trips_multi_bit_bypass() {
        let mut ops = Vec::new();
        for n in 1..=16u32 {
            let hi = (1u32 << n) - 1;
            ops.push(Op::BypassBits(n, 0));
            ops.push(Op::BypassBits(n, hi));
            ops.push(Op::BypassBits(n, hi / 3));
            ops.push(Op::Decision(n as usize % POOL, n & 1));
        }
        round_trip(&ops);
    }

    /// The property test proper: pseudo-random sequences of every op, over a
    /// pool of contexts, so states are shared between bins the way they are
    /// in a real slice.
    #[test]
    fn round_trips_random_sequences() {
        let mut seed = 0x9e3779b9u64;
        let mut lcg = || {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (seed >> 33) as u32
        };
        for trial in 0..300 {
            let len = 1 + lcg() % 400;
            let mut ops = Vec::with_capacity(len as usize);
            for _ in 0..len {
                match lcg() % 10 {
                    0..=6 => ops.push(Op::Decision((lcg() as usize) % POOL, lcg() & 1)),
                    7 | 8 => ops.push(Op::Bypass(lcg() & 1)),
                    _ => {
                        let n = 1 + lcg() % 16;
                        ops.push(Op::BypassBits(n, lcg() & ((1 << n) - 1)));
                    }
                }
            }
            // A zero terminate every so often, where a slice would have one.
            if trial % 3 == 0 {
                ops.push(Op::Terminate);
            }
            round_trip(&ops);
        }
    }
}
