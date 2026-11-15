#!/usr/bin/env python3
"""make_separate_planes.py Y.265 CB.265 CR.265 OUT.265 [--interleave]

Builds an HEVC stream coded with separate colour planes
(separate_colour_plane_flag = 1) out of three monochrome (4:0:0) streams
that were encoded with the same parameter sets and the same picture
structure — one per colour plane.

No HEVC profile allows separate_colour_plane_flag = 1 (A.3.5 to A.3.8 all
require it to be 0), so no encoder writes it and no conformance suite
carries it; but the decoding process for it is fully specified (8.1.3):
each colour plane is decoded "as if only a CVS with monochrome colour format
with that particular value of colour_plane_id would be present in the
bitstream". That is exactly what three monochrome streams are, so a stream
made from them has a known right answer: plane c of every output picture is
the corresponding picture of input stream c, which any decoder (the HM
reference decoder included) produces from that stream alone.

What changes, and nothing else does:

- the SPS: chroma_format_idc 0 becomes 3 followed by
  separate_colour_plane_flag = 1;
- every slice segment header: colour_plane_id (u(2)) after pic_output_flag
  (7.3.6.1), and in planes 1 and 2 first_slice_segment_in_pic_flag = 0 with
  an explicit slice_segment_address of 0 (only the first segment of the
  whole picture carries the flag); the header is re-aligned
  (byte_alignment()) so the slice data, and the entry point offsets that
  are relative to it, are untouched;
- the per-plane decoded picture hash SEIs become one message carrying all
  three components (D.3.20 counts them by chroma_format_idc, 3 here).

The slices of a picture are written plane by plane (all of Y, then Cb,
then Cr), or with --interleave one segment of each plane in turn — the
order 7.4.2.4.5 allows as long as each plane's own segments stay in order.

The three inputs must have byte-identical VPS / SPS / PPS and the same
sequence of slice segment NAL types and slice_pic_order_cnt_lsb; the script
checks and refuses otherwise.
"""
import argparse
import sys


class Bits:
    def __init__(self, data):
        self.d = data
        self.pos = 0

    def u(self, n):
        v = 0
        for _ in range(n):
            if self.pos >= len(self.d) * 8:
                raise EOFError("past the end of the RBSP")
            v = (v << 1) | ((self.d[self.pos >> 3] >> (7 - (self.pos & 7))) & 1)
            self.pos += 1
        return v

    def ue(self):
        z = 0
        while self.u(1) == 0:
            z += 1
        return (1 << z) - 1 + (self.u(z) if z else 0)

    def se(self):
        k = self.ue()
        return (k + 1) // 2 if k & 1 else -(k // 2)


def bits_of(data, start, end):
    """Bits [start, end) of `data` as a list of 0/1."""
    return [(data[i >> 3] >> (7 - (i & 7))) & 1 for i in range(start, end)]


def ue_bits(v):
    v += 1
    n = v.bit_length()
    return [0] * (n - 1) + [(v >> (n - 1 - i)) & 1 for i in range(n)]


def u_bits(v, n):
    return [(v >> (n - 1 - i)) & 1 for i in range(n)]


def pack(bits):
    out = bytearray()
    for i in range(0, len(bits), 8):
        chunk = bits[i:i + 8]
        chunk = chunk + [0] * (8 - len(chunk))
        b = 0
        for x in chunk:
            b = (b << 1) | x
        out.append(b)
    return bytes(out)


def unescape(nal):
    out = bytearray()
    zeros = 0
    for b in nal:
        if zeros >= 2 and b == 3:
            zeros = 0
            continue
        out.append(b)
        zeros = zeros + 1 if b == 0 else 0
    return bytes(out)


def escape(rbsp):
    out = bytearray()
    zeros = 0
    for b in rbsp:
        if zeros >= 2 and b <= 3:
            out.append(3)
            zeros = 0
        out.append(b)
        zeros = zeros + 1 if b == 0 else 0
    return bytes(out)


def nals(data):
    starts = []
    i = 0
    while True:
        j = data.find(b"\x00\x00\x01", i)
        if j < 0:
            break
        starts.append(j + 3)
        i = j + 3
    for k, s in enumerate(starts):
        e = starts[k + 1] - 3 if k + 1 < len(starts) else len(data)
        nal = data[s:e]
        while nal and nal[-1] == 0:
            nal = nal[:-1]
        if nal:
            yield nal


def payload_end(rbsp):
    """Bit position of the rbsp_stop_one_bit."""
    for i in range(len(rbsp) - 1, -1, -1):
        if rbsp[i]:
            bit = 0
            while not (rbsp[i] >> bit) & 1:
                bit += 1
            return i * 8 + (7 - bit)
    raise ValueError("no rbsp_stop_one_bit")


def ptl(b, max_sub_layers_minus1):
    b.u(8)
    b.u(32)
    b.u(48)
    b.u(8)
    present = [(b.u(1), b.u(1)) for _ in range(max_sub_layers_minus1)]
    if max_sub_layers_minus1 > 0:
        for _ in range(max_sub_layers_minus1, 8):
            b.u(2)
    for p, l in present:
        if p:
            b.u(88)
        if l:
            b.u(8)


def scaling_list_data(b):
    for size_id in range(4):
        for _ in range(0, 6, 3 if size_id == 3 else 1):
            if not b.u(1):
                b.ue()
            else:
                if size_id > 1:
                    b.se()
                for _ in range(min(64, 1 << (4 + (size_id << 1)))):
                    b.se()


def st_rps(b, idx, num, counts):
    """st_ref_pic_set(idx); returns NumDeltaPocs of the set."""
    inter = b.u(1) if idx != 0 else 0
    if inter:
        delta_idx = b.ue() + 1 if idx == num else 1
        ref = counts[idx - delta_idx]
        b.u(1)
        b.ue()
        kept = 0
        for _ in range(ref + 1):
            used = b.u(1)
            use_delta = 1 if used else b.u(1)
            kept += use_delta
        return kept
    neg = b.ue()
    pos = b.ue()
    for _ in range(neg + pos):
        b.ue()
        b.u(1)
    return neg + pos


def parse_sps(rb):
    """SPS fields a slice header needs, and the bit position of
    chroma_format_idc."""
    b = Bits(rb)
    b.u(4)
    msl = b.u(3)
    b.u(1)
    ptl(b, msl)
    b.ue()
    cf_pos = b.pos
    chroma = b.ue()
    cf_end = b.pos
    sep = b.u(1) if chroma == 3 else 0
    w = b.ue()
    h = b.ue()
    if b.u(1):
        for _ in range(4):
            b.ue()
    b.ue()
    b.ue()
    log2_poc = b.ue() + 4
    info = b.u(1)
    for _ in range(0 if info else msl, msl + 1):
        b.ue(), b.ue(), b.ue()
    log2_min_cb = b.ue() + 3
    log2_ctb = log2_min_cb + b.ue()
    b.ue(), b.ue(), b.ue(), b.ue()
    if b.u(1) and b.u(1):
        scaling_list_data(b)
    b.u(1)
    sao = b.u(1)
    if b.u(1):
        b.u(4), b.u(4), b.ue(), b.ue(), b.u(1)
    num_st = b.ue()
    counts = []
    for i in range(num_st):
        counts.append(st_rps(b, i, num_st, counts))
    lt_present = b.u(1)
    num_lt_sps = 0
    if lt_present:
        num_lt_sps = b.ue()
        for _ in range(num_lt_sps):
            b.u(log2_poc)
            b.u(1)
    tmvp = b.u(1)
    ctb = 1 << log2_ctb
    n_ctbs = ((w + ctb - 1) // ctb) * ((h + ctb - 1) // ctb)
    return dict(chroma=chroma, sep=sep, cf_pos=cf_pos, cf_end=cf_end, log2_poc=log2_poc,
                num_st=num_st, counts=counts, lt_present=lt_present, num_lt_sps=num_lt_sps,
                tmvp=tmvp, sao=sao, addr_bits=(n_ctbs - 1).bit_length())


def parse_pps(rb):
    b = Bits(rb)
    b.ue()
    b.ue()
    p = dict(dep=b.u(1), output_flag=b.u(1), extra=b.u(3))
    b.u(1)
    p["cabac_init_present"] = b.u(1)
    p["l0_default"] = b.ue() + 1
    p["l1_default"] = b.ue() + 1
    b.se()
    b.u(1)
    b.u(1)
    if b.u(1):
        b.ue()
    b.se(), b.se()
    p["slice_chroma_qp"] = b.u(1)
    p["wp"] = b.u(1)
    p["wbi"] = b.u(1)
    b.u(1)
    tiles = b.u(1)
    wpp = b.u(1)
    p["entry_points"] = tiles or wpp
    if tiles:
        cols = b.ue() + 1
        rows = b.ue() + 1
        if not b.u(1):
            for _ in range(cols - 1 + rows - 1):
                b.ue()
        b.u(1)
    p["lf_across_slices"] = b.u(1)
    p["dbk_override"] = 0
    p["dbk_disabled"] = 0
    if b.u(1):
        p["dbk_override"] = b.u(1)
        p["dbk_disabled"] = b.u(1)
        if not p["dbk_disabled"]:
            b.se(), b.se()
    if b.u(1):
        scaling_list_data(b)
    p["lists_mod"] = b.u(1)
    b.ue()
    p["header_ext"] = b.u(1)
    p["ext"] = b.u(1)
    return p


def slice_header_end(rb, nal_type, sps, pps):
    """(position of first_slice_segment_in_pic_flag's end, position after
    slice_pic_parameter_set_id, where colour_plane_id goes, end of the
    header before byte_alignment(), dependent, poc_lsb)."""
    b = Bits(rb)
    first = b.u(1)
    if 16 <= nal_type <= 23:
        b.u(1)
    b.ue()
    after_pps_id = b.pos
    dependent = 0
    if not first:
        if pps["dep"]:
            dependent = b.u(1)
        b.u(sps["addr_bits"])
    cp_pos = None
    poc = None
    slice_type = None
    sao_luma = sao_chroma = 0
    dbk_disabled = pps["dbk_disabled"]
    if not dependent:
        b.u(pps["extra"])
        slice_type = b.ue()
        if pps["output_flag"]:
            b.u(1)
        cp_pos = b.pos
        tmvp = 0
        if nal_type not in (19, 20):
            poc = b.u(sps["log2_poc"])
            if not b.u(1):
                st_rps(b, sps["num_st"], sps["num_st"], sps["counts"])
            elif sps["num_st"] > 1:
                b.u((sps["num_st"] - 1).bit_length())
            if sps["lt_present"]:
                n_sps = b.ue() if sps["num_lt_sps"] > 0 else 0
                n_pics = b.ue()
                for i in range(n_sps + n_pics):
                    if i < n_sps:
                        if sps["num_lt_sps"] > 1:
                            b.u((sps["num_lt_sps"] - 1).bit_length())
                    else:
                        b.u(sps["log2_poc"])
                        b.u(1)
                    if b.u(1):
                        b.ue()
            if sps["tmvp"]:
                tmvp = b.u(1)
        if sps["sao"]:
            sao_luma = b.u(1)
            # ChromaArrayType is 0 in every stream this script reads or writes.
        if slice_type in (0, 1):
            l0, l1 = pps["l0_default"], pps["l1_default"] if slice_type == 0 else 0
            if b.u(1):
                l0 = b.ue() + 1
                if slice_type == 0:
                    l1 = b.ue() + 1
            if pps["lists_mod"]:
                raise SystemExit("lists_modification_present_flag is not handled")
            if slice_type == 0:
                b.u(1)
            if pps["cabac_init_present"]:
                b.u(1)
            if tmvp:
                col_l0 = b.u(1) if slice_type == 0 else 1
                if (col_l0 and l0 > 1) or (not col_l0 and l1 > 1):
                    b.ue()
            if (pps["wp"] and slice_type == 1) or (pps["wbi"] and slice_type == 0):
                raise SystemExit("weighted prediction is not handled")
            b.ue()
        b.se()
        if pps["slice_chroma_qp"]:
            b.se(), b.se()
        override = b.u(1) if pps["dbk_override"] else 0
        if override:
            dbk_disabled = b.u(1)
            if not dbk_disabled:
                b.se(), b.se()
        if pps["lf_across_slices"] and (sao_luma or sao_chroma or not dbk_disabled):
            b.u(1)
    if pps["entry_points"]:
        n = b.ue()
        if n:
            ln = b.ue() + 1
            for _ in range(n):
                b.u(ln)
    if pps["header_ext"]:
        raise SystemExit("slice_segment_header_extension is not handled")
    end = b.pos
    # byte_alignment(): a one, then zeros to the byte boundary.
    if b.u(1) != 1:
        raise SystemExit("slice header did not end where it should")
    while b.pos % 8:
        if b.u(1):
            raise SystemExit("slice header alignment bits are not zero")
    return dict(first=first, after_pps_id=after_pps_id, cp_pos=cp_pos, end=end,
                data=b.pos // 8, dependent=dependent, poc=poc, slice_type=slice_type)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("y")
    ap.add_argument("cb")
    ap.add_argument("cr")
    ap.add_argument("out")
    ap.add_argument("--interleave", action="store_true")
    a = ap.parse_args()
    streams = [list(nals(open(p, "rb").read())) for p in (a.y, a.cb, a.cr)]
    ps = [[n for n in s if 32 <= (n[0] >> 1) & 0x3F <= 34] for s in streams]
    if not (ps[0] == ps[1] == ps[2]):
        raise SystemExit("the three streams' parameter sets differ")
    sps = pps = None
    out = bytearray()
    # Each picture: its slice segments per plane, and its hash per plane.
    pending = None

    def flush():
        if pending is None:
            return
        segs, hashes = pending
        if len({len(s) for s in segs}) != 1 and a.interleave:
            raise SystemExit("--interleave needs the same number of segments per plane")
        order = []
        if a.interleave:
            for i in range(len(segs[0])):
                order += [segs[c][i] for c in range(3)]
        else:
            order = segs[0] + segs[1] + segs[2]
        for n in order:
            out.extend(b"\x00\x00\x00\x01" + n)
        if all(h is not None for h in hashes):
            kinds = {h[0] for h in hashes}
            if len(kinds) == 1:
                k = kinds.pop()
                payload = bytes([k]) + b"".join(h[1] for h in hashes)
                sei = bytes([132, len(payload)]) + payload + b"\x80"
                out.extend(b"\x00\x00\x00\x01" + bytes([0x50, 0x01]) + escape(sei))

    iters = [iter(s) for s in streams]
    # Walk the three streams NAL by NAL in lockstep: they have the same
    # structure, so the k-th NAL of each is the same kind of thing.
    for trio in zip(*iters):
        t = [(n[0] >> 1) & 0x3F for n in trio]
        if len(set(t)) != 1:
            raise SystemExit(f"the streams' NAL types differ: {t}")
        t = t[0]
        if t in (32, 34, 35, 39):
            # VPS, PPS, AUD, prefix SEI: plane 0's copy (the parameter sets
            # are identical; the others belong to the picture, not a plane).
            flush()
            pending = None
            if t == 34:
                pps = parse_pps(unescape(trio[0][2:]))
            out.extend(b"\x00\x00\x00\x01" + trio[0])
            continue
        if t == 33:
            flush()
            pending = None
            rb = unescape(trio[0][2:])
            sps = parse_sps(rb)
            if sps["chroma"] != 0:
                raise SystemExit("the inputs must be monochrome")
            end = payload_end(rb)
            bits = bits_of(rb, 0, sps["cf_pos"]) + ue_bits(3) + [1] + bits_of(rb, sps["cf_end"], end)
            bits += [1]
            out.extend(b"\x00\x00\x00\x01" + trio[0][:2] + escape(pack(bits)))
            continue
        if t == 40:
            # Suffix SEI: the per-plane decoded picture hashes.
            if pending is not None:
                for c, n in enumerate(trio):
                    rb = unescape(n[2:])
                    if rb[0] == 132:
                        size = rb[1]
                        body = rb[2:2 + size]
                        pending[1][c] = (body[0], body[1:])
            continue
        if t > 31:
            continue
        # A slice segment of each plane.
        segs = []
        for c, n in enumerate(trio):
            rb = unescape(n[2:])
            h = slice_header_end(rb, t, sps, pps)
            segs.append((n[:2], rb, h))
        if segs[0][2]["first"]:
            flush()
            pending = ([[], [], []], [None, None, None])
        pocs = {s[2]["poc"] for s in segs}
        if len(pocs) != 1:
            raise SystemExit(f"the streams' POCs differ: {pocs}")
        for c, (hdr2, rb, h) in enumerate(segs):
            bits = []
            if h["first"] and c > 0:
                bits += [0] + bits_of(rb, 1, h["after_pps_id"])
                if pps["dep"]:
                    bits += [0]
                bits += u_bits(0, sps["addr_bits"])
                bits += bits_of(rb, h["after_pps_id"], h["cp_pos"])
            elif h["cp_pos"] is not None:
                bits += bits_of(rb, 0, h["cp_pos"])
            else:
                bits += bits_of(rb, 0, h["end"])
            if h["cp_pos"] is not None:
                bits += u_bits(c, 2)
                bits += bits_of(rb, h["cp_pos"], h["end"])
            bits += [1]
            while len(bits) % 8:
                bits.append(0)
            new = pack(bits) + rb[h["data"]:]
            pending[0][c].append(hdr2 + escape(new))
    flush()
    open(a.out, "wb").write(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
