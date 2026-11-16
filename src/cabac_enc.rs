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
use crate::h264::cabac_mb::NUM_CTX;

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
/// `rangeTabLps` by the context's whole state byte (`pStateIdx << 1 |
/// valMPS`), its four columns (`(codIRange >> 6) & 3`) as the bytes of one
/// word, lowest first: [`WCore::decision`]'s table.
static LPS_WORD: [u32; 128] = {
    let mut t = [0u32; 128];
    let mut c = 0;
    while c < 128 {
        t[c] = u32::from_le_bytes(LPS_RANGE[c >> 1]);
        c += 1;
    }
    t
};

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
/// the bits that leave its ten-bit window above it, and they go out
/// thirty-two at a time into a byte buffer of the writer's own, a carry
/// out of them incrementing the bytes already there (the last, and the
/// `0xff` run before it, which it turns to `0x00`) — the same codeword,
/// since the outstanding bits are exactly the run a carry would flip. The
/// buffer joins the caller's [`BitWriter`] when the slice ends. The
/// module's round trip holds the two coders to the same bytes.
///
/// A word at a time, a bin checks a threshold it crosses about once in
/// thirty bins, and what it then does has no branch on the data but the
/// carry's (rare): a byte at a time, with the pending byte and its `0xff`
/// run kept apart for a carry, the check was crossed every eight or so
/// bins at a point the data decides, and its out-of-line path cost more
/// than the bins between.
pub(crate) struct CabacWriter<'a> {
    w: &'a mut BitWriter,
    /// The registers every bin touches, kept apart so a hot loop can hold
    /// them by value ([`BinEnc::split`]).
    core: WCore,
    /// The codeword's bytes so far, all but the first bit
    /// (`firstBitFlag`'s), which ...
    buf: Vec<u8>,
    /// ... is still among the queued bits while this is 1.
    lead: u32,
}

/// [`CabacWriter`]'s per-bin state.
#[derive(Clone, Copy)]
pub(crate) struct WCore {
    /// `codILow`, with [`WCore::queued`] bits above its window (and, past
    /// those, a carry not yet handed on).
    low: u64,
    /// `codIRange`.
    range: u32,
    /// How many bits `low` holds above its ten-bit window.
    queued: u32,
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
            },
            buf: Vec::with_capacity(1 << 12),
            lead: 1,
        }
    }

    /// One added to the bytes written: a carry out of the queued bits.
    #[cold]
    #[inline(never)]
    fn carry(&mut self) {
        for b in self.buf.iter_mut().rev() {
            *b = b.wrapping_add(1);
            if *b != 0 {
                return;
            }
        }
        debug_assert!(false, "a carry past the codeword's first bit");
    }

    /// [`WCore::settle`]'s word: the top thirty-two queued bits (and, the
    /// first time, the codeword's first bit above them, dropped) out to
    /// the buffer, a carry above them into the bytes before.
    #[inline(always)]
    fn word_out(&mut self, low: u64, queued: u32) -> (u64, u32) {
        let take = 32 + self.lead;
        let shift = 10 + queued - take;
        let v = low >> shift;
        debug_assert!(
            self.lead == 0 || v >> take == 0,
            "a carry past the codeword's first bit"
        );
        if v >> take != 0 {
            self.carry();
        }
        self.buf.extend_from_slice(&(v as u32).to_be_bytes());
        self.lead = 0;
        (low & ((1 << shift) - 1), queued - take)
    }

    /// [`CabacWriter`]'s `EncodeFlush`: the window's top three bits, the
    /// last forced to one, join the queue, everything goes to the buffer
    /// and the buffer to the writer, the bits short of a byte last.
    fn fast_flush(&mut self) {
        let mut c = self.core;
        c.range = 2;
        c.renorm(self);
        c.low |= 0x80;
        c.low <<= 3;
        c.queued += 3;
        // Whole bytes, the first (when nothing has gone yet) with the
        // leading bit above it.
        while c.queued >= 8 + self.lead {
            let need = 8 + self.lead;
            let shift = 10 + c.queued - need;
            let v = c.low >> shift;
            c.low &= (1 << shift) - 1;
            c.queued -= need;
            if v >> need != 0 {
                self.carry();
            }
            self.buf.push(v as u8);
            self.lead = 0;
        }
        let mut rest = c.queued;
        let mut v = (c.low >> 10) as u32;
        if self.lead == 1 {
            // Never reached a byte: drop the leading bit here.
            debug_assert!(rest >= 1);
            rest -= 1;
            v &= (1 << rest) - 1;
            self.lead = 0;
        }
        // A carry out of the last bits still reaches the bytes.
        if v >> rest != 0 {
            self.carry();
        }
        let w = &mut *self.w;
        for &b in &self.buf {
            w.bits(8, b as u32);
        }
        self.buf.clear();
        if rest > 0 {
            w.bits(rest, v & ((1 << rest) - 1));
        }
        c.queued = 0;
        c.low = 0;
        self.core = c;
    }
}

impl WCore {
    /// How many bits may wait above the window before a word goes out
    /// (thirty-two of them, or thirty-three the first time): with the
    /// eight a bin can add and the carry above them, the window's ten and
    /// the queue stay inside `low`'s sixty-four.
    const BATCH: u32 = 33;

    /// Move a word from above the window to the buffer, when enough bits
    /// have gathered.
    #[inline(always)]
    fn settle(&mut self, out: &mut CabacWriter<'_>) {
        if self.queued >= Self::BATCH {
            (self.low, self.queued) = out.word_out(self.low, self.queued);
        }
    }

    /// The renormalisation: one shift, then a word out when there is one.
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
        let mps = (state & 1) as u32;
        // The context's four LPS sub-ranges are one word, loaded from the
        // state alone; the range only picks a byte of it by a shift — off
        // the range's dependency chain, where a table load indexed by the
        // range would sit on it.
        let lps = (LPS_WORD[state as usize & 127] >> ((self.range >> 3) & 0x18)) & 0xff;
        // Whether a bin is the LPS is the data's business, so the update
        // and the renormalisation are selects and an unconditional shift
        // (by zero when the range is still in its upper half: a nine-bit
        // range has 23 leading zeros) rather than branches.
        let rmps = self.range - lps;
        let is_lps = bin != mps;
        self.low += if is_lps { rmps as u64 } else { 0 };
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
        self.low = (self.low << 1) + (self.range & 0u32.wrapping_sub((bin != 0) as u32)) as u64;
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
            self.core.low += self.core.range as u64;
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
    /// A context-coded bin only when `cond` holds — branch-free where the
    /// coder can make it so (the estimator's).
    #[inline(always)]
    fn decision_if(&mut self, out: &mut O, ctx: &mut Ctx, bin: u32, cond: bool) {
        if cond {
            self.decision(out, ctx, bin);
        }
    }
    /// `n` (at most [`ONES_MAX`]) context-coded ones in a row on one
    /// context: a level's unary prefix.
    #[inline(always)]
    fn ones(&mut self, out: &mut O, ctx: &mut Ctx, n: u32) {
        for _ in 0..n {
            self.decision(out, ctx, 1);
        }
    }
    /// A significance map whose positions each have their own
    /// significance and last contexts (`sig + i`, `last + i`): for each
    /// position before `end`, its bit of `mask`, then — where that is set
    /// — whether it is `top`. The walk a position at a time; the
    /// estimator's prices every position at once.
    #[allow(clippy::too_many_arguments)]
    #[inline(always)]
    fn sig_map_distinct(
        &mut self,
        out: &mut O,
        ctx: &mut [Ctx; NUM_CTX],
        sig: usize,
        last: usize,
        mask: u64,
        top: usize,
        end: usize,
    ) {
        sig_map_walk(self, out, ctx, sig, last, mask, top, end);
    }
}

/// [`BinCore::sig_map_distinct`] a position at a time, each last bin
/// priced or not by a mask rather than a branch ([`BinCore::decision_if`]).
#[allow(clippy::too_many_arguments)]
#[inline(always)]
fn sig_map_walk<O: ?Sized, C: BinCore<O>>(
    c: &mut C,
    out: &mut O,
    ctx: &mut [Ctx; NUM_CTX],
    sig: usize,
    last: usize,
    mask: u64,
    top: usize,
    end: usize,
) {
    const M: usize = NUM_CTX - 1;
    for i in 0..end {
        let b = ((mask >> i) & 1) as u32;
        c.decision(out, &mut ctx[(sig + i) & M], b);
        c.decision_if(out, &mut ctx[(last + i) & M], (i == top) as u32, b != 0);
    }
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
    /// The coder only prices bins ([`CabacEstimator`]): the residual
    /// writer then walks a block without branching on its values, which
    /// a coder writing bytes cannot do and need not (it codes each block
    /// once, the estimator every candidate's).
    const PRICES: bool = false;
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
/// the same context transitions, with no output to dispatch on — and the
/// total kept in fixed point ([`EST_ONE`] to the bit), so that a bin is an
/// integer add rather than a floating one whose latency every bin of a
/// block waits on.
#[derive(Default)]
pub(crate) struct CabacEstimator {
    acc: u64,
}

/// One bit in [`CabacEstimator`]'s fixed point.
const EST_ONE: u64 = 1 << 16;

/// [`EST_COST`] in [`CabacEstimator`]'s fixed point, rounded.
static EST_COST_FX: [u32; 256] = {
    let mut t = [0u32; 256];
    let mut i = 0;
    while i < 256 {
        t[i] = (EST_COST[i] as f64 * EST_ONE as f64 + 0.5) as u32;
        i += 1;
    }
    t
};

/// [`EST_COST_FX`] and [`EST_NEXT`] by a bin's *key*, `state ^ bin` —
/// `pStateIdx << 1` with whether the bin is the LPS, all a bin's price
/// depends on — as four 128-byte planes for byte table lookups: the
/// price's three bytes, lowest first, and the next state with `valMPS`
/// still to be xored in (a state's next state is the same bits whatever
/// its `valMPS`, but for that bit). [`sig_map_neon`]'s tables.
#[cfg(target_arch = "aarch64")]
static KEY_PLANES: [[u8; 128]; 4] = {
    let mut t = [[0u8; 128]; 4];
    let mut k = 0;
    while k < 128 {
        // The state with valMPS 0 and the bin that gives this key.
        let i = (((k & !1) << 1) | (k & 1)) & 255;
        let c = EST_COST_FX[i];
        t[0][k] = c as u8;
        t[1][k] = (c >> 8) as u8;
        t[2][k] = (c >> 16) as u8;
        t[3][k] = EST_NEXT[i];
        k += 1;
    }
    t
};

/// Whether [`EstCore::sig_map_distinct`] takes [`sig_map_neon`]: where
/// the `h264_enc` table would take NEON (not under `H26X_NO_SIMD`,
/// `H26X_MAX_SIMD=none` or `H26X_ENC_NO_SIMD`), decided once.
#[cfg(target_arch = "aarch64")]
fn sig_map_simd() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        crate::dsp::Cpu::detect_honouring_env().neon && !crate::dsp::enc_simd_disabled("h264_enc")
    })
}

/// [`BinCore::sig_map_distinct`] for the estimator in NEON: every
/// position's significance bin and (where significant) last bin at once,
/// sixteen lanes of each. The positions' contexts are distinct and none
/// is read after another is written, so the bins are independent: each
/// lane's state moves as the bin-at-a-time walk moves it, and the price
/// is the same integers summed in another order. `sig` and `last` are
/// the sixteen states from the block's first significance and last
/// context; lanes from `end` on are left as they are. Returns the price.
///
/// # Safety
/// NEON; `sig` and `last` each sixteen bytes, not overlapping.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn sig_map_neon(sig: *mut u8, last: *mut u8, mask: u64, top: usize, end: usize) -> u64 {
    use std::arch::aarch64::*;
    unsafe {
        const IOTA: [u8; 16] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];
        const BIT: [u8; 16] = [1, 2, 4, 8, 16, 32, 64, 128, 1, 2, 4, 8, 16, 32, 64, 128];
        let iota = vld1q_u8(IOTA.as_ptr());
        let one = vdupq_n_u8(1);
        // Positions before `end`; the significant ones, as 0xff bytes.
        let on = vcltq_u8(iota, vdupq_n_u8(end as u8));
        let m = vcombine_u8(vdup_n_u8(mask as u8), vdup_n_u8((mask >> 8) as u8));
        let nz = vtstq_u8(m, vld1q_u8(BIT.as_ptr()));
        let lon = vandq_u8(on, nz);
        let s = vld1q_u8(sig);
        let l = vld1q_u8(last);
        let ks = veorq_u8(s, vandq_u8(nz, one));
        let kl = veorq_u8(l, vandq_u8(vceqq_u8(iota, vdupq_n_u8(top as u8)), one));
        let p = KEY_PLANES.as_ptr() as *const u8;
        let (ks_hi, kl_hi) = (vsubq_u8(ks, vdupq_n_u8(64)), vsubq_u8(kl, vdupq_n_u8(64)));
        // A 128-byte plane looked up: its low half, then its high half
        // where the key is past it (out of range in the first, kept).
        let look = |plane: usize, k: uint8x16_t, k_hi: uint8x16_t| -> uint8x16_t {
            let base = p.add(128 * plane);
            vqtbx4q_u8(
                vqtbl4q_u8(vld1q_u8_x4(base), k),
                vld1q_u8_x4(base.add(64)),
                k_hi,
            )
        };
        let ns = veorq_u8(look(3, ks, ks_hi), vandq_u8(s, one));
        let nl = veorq_u8(look(3, kl, kl_hi), vandq_u8(l, one));
        vst1q_u8(sig, vbslq_u8(on, ns, s));
        vst1q_u8(last, vbslq_u8(lon, nl, l));
        let mut sum = 0u64;
        for b in 0..3 {
            let c = vaddq_u16(
                vpaddlq_u8(vandq_u8(look(b, ks, ks_hi), on)),
                vpaddlq_u8(vandq_u8(look(b, kl, kl_hi), lon)),
            );
            sum += (vaddvq_u16(c) as u64) << (8 * b);
        }
        sum
    }
}

impl CabacEstimator {
    /// A fresh estimate.
    pub(crate) fn new() -> Self {
        Self::default()
    }

    /// The bins' estimated cost so far, in bits.
    pub(crate) fn fractional_bits(&self) -> f64 {
        self.acc as f64 / EST_ONE as f64
    }
}

/// [`CabacEstimator`]'s [`BinCore`]: the running total.
#[derive(Clone, Copy)]
pub(crate) struct EstCore(u64);

/// The most ones [`BinCore::ones`] codes at once: a level's prefix past
/// its first bin (thirteen at the cap of fourteen).
pub(crate) const ONES_MAX: u32 = 13;

/// `n` ones in a row from each context state, priced and moved:
/// `ONES[state][n]` is the price of the run in [`CabacEstimator`]'s fixed
/// point, shifted up a byte, with the state the run leaves in the low
/// byte — what `n` of [`EstCore::decision`] add and leave, one dependent
/// load where the bins were a chain of `n` (each state the one before's
/// table entry, through memory).
static ONES: [[u32; ONES_MAX as usize + 1]; 128] = {
    let mut t = [[0u32; ONES_MAX as usize + 1]; 128];
    let mut s = 0;
    while s < 128 {
        let (mut st, mut cost) = (s, 0u32);
        let mut n = 0;
        while n <= ONES_MAX as usize {
            t[s][n] = (cost << 8) | st as u32;
            let i = (st << 1) | 1;
            cost += EST_COST_FX[i];
            st = EST_NEXT[i] as usize;
            n += 1;
        }
        s += 1;
    }
    t
};

impl BinCore<CabacEstimator> for EstCore {
    #[inline(always)]
    fn decision(&mut self, _: &mut CabacEstimator, ctx: &mut Ctx, bin: u32) {
        let i = ((*ctx as usize) << 1 | bin as usize) & 255;
        self.0 += EST_COST_FX[i] as u64;
        *ctx = EST_NEXT[i];
    }
    #[inline(always)]
    fn ones(&mut self, _: &mut CabacEstimator, ctx: &mut Ctx, n: u32) {
        debug_assert!(n <= ONES_MAX);
        let e = ONES[*ctx as usize & 127][n.min(ONES_MAX) as usize];
        self.0 += (e >> 8) as u64;
        *ctx = e as u8;
    }
    #[inline(always)]
    fn bypass(&mut self, _: &mut CabacEstimator, _bin: u32) {
        self.0 += EST_ONE;
    }
    #[inline(always)]
    fn decision_if(&mut self, _: &mut CabacEstimator, ctx: &mut Ctx, bin: u32, cond: bool) {
        let i = ((*ctx as usize) << 1 | bin as usize) & 255;
        // The bin's cost masked in, the state's move selected: no branch.
        self.0 += (EST_COST_FX[i] & 0u32.wrapping_sub(cond as u32)) as u64;
        *ctx = if cond { EST_NEXT[i] } else { *ctx };
    }
    #[inline(always)]
    fn sig_map_distinct(
        &mut self,
        out: &mut CabacEstimator,
        ctx: &mut [Ctx; NUM_CTX],
        sig: usize,
        last: usize,
        mask: u64,
        top: usize,
        end: usize,
    ) {
        #[cfg(target_arch = "aarch64")]
        if sig_map_simd() && sig + 16 <= NUM_CTX && last + 16 <= NUM_CTX && sig.abs_diff(last) >= 16
        {
            debug_assert!(end <= 15 && top < 16 && mask < 1 << 16);
            let p = ctx.as_mut_ptr();
            // SAFETY: NEON is baseline here; both sixteen-byte windows are
            // inside the array (checked) and apart.
            self.0 += unsafe { sig_map_neon(p.add(sig), p.add(last), mask, top, end) };
            return;
        }
        sig_map_walk(self, out, ctx, sig, last, mask, top, end);
    }
}

impl BinEnc for CabacEstimator {
    type Core = EstCore;
    type Out = CabacEstimator;
    const PRICES: bool = true;
    #[inline(always)]
    fn split(&mut self) -> (EstCore, &mut Self) {
        (EstCore(self.acc), self)
    }
    #[inline(always)]
    fn join(&mut self, core: EstCore) {
        self.acc = core.0;
    }
    #[inline(always)]
    fn encode_decision(&mut self, ctx: &mut Ctx, bin: u32) {
        let i = ((*ctx as usize) << 1 | bin as usize) & 255;
        self.acc += EST_COST_FX[i] as u64;
        *ctx = EST_NEXT[i];
    }
    #[inline(always)]
    fn encode_bypass(&mut self, _bin: u32) {
        self.acc += EST_ONE;
    }
    #[inline(always)]
    fn encode_bypass_bits(&mut self, n: u32, _v: u32) {
        self.acc += n as u64 * EST_ONE;
    }
    #[inline(always)]
    fn encode_terminate(&mut self, bin: u32) {
        // 7 bits and 0.01 of one, as the floating coders charge them.
        self.acc += if bin != 0 { 7 * EST_ONE } else { EST_ONE / 100 };
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

    /// The estimator's run of ones from its table is the bins one at a
    /// time: the same price and the same state, from every state and
    /// every run length.
    #[test]
    fn estimator_ones_are_the_bins() {
        for s in 0..128u8 {
            for n in 0..=ONES_MAX {
                let mut est = CabacEstimator::new();
                let (mut c, out) = est.split();
                let mut a = s;
                c.ones(out, &mut a, n);
                let mut walk = EstCore(0);
                let mut b = s;
                for _ in 0..n {
                    walk.decision(&mut CabacEstimator::new(), &mut b, 1);
                }
                assert_eq!((c.0, a), (walk.0, b), "state {s} run {n}");
            }
        }
    }

    /// The estimator's whole-map pricing ([`EstCore::sig_map_distinct`],
    /// NEON where there is one) against the walk a position at a time
    /// ([`sig_map_walk`]): the same price and the same context array, over
    /// random states, masks and block lengths, at every category's bases.
    #[test]
    fn estimator_sig_map_is_the_walk() {
        let mut seed = 0x2545f4914f6cdd1du64;
        let mut lcg = || {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (seed >> 33) as u32
        };
        let bases = [
            (105, 166),
            (120, 181),
            (152, 213),
            (484, 572),
            (849, 937),
            (1000, 900),
        ];
        for trial in 0..20000 {
            let mut ctx = [0u8; NUM_CTX];
            for v in ctx.iter_mut() {
                *v = (lcg() % 126) as u8;
            }
            let (sig, last) = bases[trial % bases.len()];
            let n = [4usize, 15, 16][trial % 3];
            let mask = (lcg() as u64 & ((1 << n) - 1)) | (1 << (lcg() as usize % n));
            let top = 63 - mask.leading_zeros() as usize;
            let end = if top == n - 1 { n - 1 } else { top + 1 };
            let mut a = ctx;
            let mut est = CabacEstimator::new();
            let (mut c, out) = est.split();
            c.sig_map_distinct(out, &mut a, sig, last, mask, top, end);
            let mut b = ctx;
            let mut est2 = CabacEstimator::new();
            let (mut w, out2) = est2.split();
            sig_map_walk(&mut w, out2, &mut b, sig, last, mask, top, end);
            assert_eq!(c.0, w.0, "trial {trial}: price");
            assert!(a == b, "trial {trial}: states");
        }
    }

    /// [`KEY_PLANES`] is [`EST_COST_FX`] and [`EST_NEXT`] by key, from
    /// either `valMPS`.
    #[cfg(target_arch = "aarch64")]
    #[test]
    fn key_planes_are_the_tables() {
        for s in 0..128usize {
            for bin in 0..2usize {
                let i = (s << 1) | bin;
                let k = s ^ bin;
                let c = KEY_PLANES[0][k] as u32
                    | (KEY_PLANES[1][k] as u32) << 8
                    | (KEY_PLANES[2][k] as u32) << 16;
                assert_eq!(c, EST_COST_FX[i], "state {s} bin {bin}");
                assert_eq!(KEY_PLANES[3][k] ^ (s & 1) as u8, EST_NEXT[i]);
            }
        }
    }
}
