// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// Q3 GQA-aware systolic sparse attention, shared by
// SparsePA (QSA_ADDRESS_MODE 0) and SparseSDPA (QSA_ADDRESS_MODE 1). With QSA_GQA_SPLIT the block steps of a token
// are partitioned over QSA_SPLIT_PARTITIONS work-groups that emit partial (max, sum, acc) state for
// sparse_attention_split_reduce.cl.
//
// One work-group = 4 subgroups = ONE token x ONE KV head. A KV step is 16 rows (16 / r selected blocks); each subgroup
// owns a Dh / 4 slice of the QK reduction and a Dv / 4 slice of the output.
//   QK: A = K rows of the step (2D block reads), B = Q with lane = query head of the KV group (the remaining lanes
//       carry zero Q); the score tile has lane = head, so the softmax is in-lane. Worker partials meet in SLM.
//   PV: A = P with lane = KV row (transposed through a per-worker SLM tile), B = V in VNNI form, so the accumulator
//       has element = head, lane = output column.
// Valid rows of a step are always the prefix [0, valid_kv), made of page-local runs of R rows. A step that is one
// contiguous 16-row run is read with one 2D message per 32 columns (VNNI transform for V), otherwise with one message
// per block and 32 columns, all issued before any is consumed (rows past a short block read as zero; V is packed to
// VNNI in registers). The 2D path needs 64-byte aligned K/V buffers; without 2D block IO rows are
// read one by one.
// p stays below 2^8 in half thanks to a lazy rescale of the running max (scores are in log2 units).
//
// Quantized K/V (paged u8 pages or dense i8 tensors with separate scales / zero points, see qsa_common.cl) are
// dequantized to half in registers, so the dpas operands are identical to the f16 ones. With 2D block IO and 64-byte
// rows (GQA_KQ2D / GQA_VQ2D) the codes of a step are read with one 8-bit 2D message per block and pair of 32-byte
// chunks: a lane then holds 2 bytes of a chunk, i.e. E = 2 (8-bit) or 4 (4-bit) adjacent channels, and tile q of
// chunk c is channel c * 16 E + E * lane + q. K uses this channel order for the QK reduction (Q is loaded permuted),
// V gives permuted output columns that the final stores put back in order. Otherwise quantized rows are read one by
// one.
//
// JIT: QSA_HEADS, QSA_KV_HEADS, QSA_K_HEAD_SIZE, QSA_V_HEAD_SIZE, QSA_COMPRESS_RATIO, QSA_BLOCK_TOPK, QSA_SCALE,
//      QSA_IS_CAUSAL, QSA_ADDRESS_MODE, SUBGROUP_SIZE, (split) QSA_GQA_SPLIT, QSA_SPLIT_PARTITIONS, (paged) QSA_PAGE_SIZE,
//      QSA_{K,V}_{MODE,BITS,SYM,GROUP,ROW_BYTES,PAGE_BYTES,PARAM_OFFSET}, (dense) QSA_DENSE_QUANT,
//      QSA_{K,V}_{SIGNED,ZP_F16}
// Dispatch: gws = [tokens * 4 * SG, QSA_KV_HEADS, P (split) | 1], lws = [4 * SG, 1, 1]

#define QSA_XMX_WORKERS 4
#define QSA_ATTENTION_HELPERS
#include "qsa_common.cl"

#ifdef cl_intel_subgroups_long
#    pragma OPENCL EXTENSION cl_intel_subgroups_long : enable
#endif

#if SUBGROUP_SIZE != 16
#    error "the systolic attention needs a 16-wide subgroup"
#endif
#if !QSA_IS_CAUSAL
#    error "only causal QSA is supported"
#endif
#ifndef QSA_GQA_SPLIT
#    define QSA_GQA_SPLIT 0
#endif
#if QSA_COMPRESS_RATIO != 2 && QSA_COMPRESS_RATIO != 4 && QSA_COMPRESS_RATIO != 8 && QSA_COMPRESS_RATIO != 16
#    error "R must be 2, 4, 8 or 16"
#endif
#if (QSA_K_HEAD_SIZE % (QSA_XW * 32)) != 0 || (QSA_V_HEAD_SIZE % (QSA_XW * 32)) != 0
#    error "worker slices must be 32-half aligned"
#endif

#define GQA_R QSA_COMPRESS_RATIO
#define GQA_BLOCKS (16 / QSA_COMPRESS_RATIO)
#define GQA_RESCALE_LOG2 8.0f
#define GQA_VT (QSA_XDV / 16)  // 16-column PV tiles of a worker
#define GQA_HEADS (QSA_HEADS / QSA_KV_HEADS)
#define GQA_HH (GQA_HEADS > 8 ? 2 : 1)  // 8-head halves of the PV accumulator

#undef GQA_BLOCK_READ
#ifdef cl_intel_subgroup_2d_block_io
#    define GQA_2D 1
#    if QSA_COMPRESS_RATIO == 2
#        define GQA_BLOCK_READ intel_sub_group_2d_block_read_16b_2r16x2c
#    elif QSA_COMPRESS_RATIO == 4
#        define GQA_BLOCK_READ intel_sub_group_2d_block_read_16b_4r16x2c
#    elif QSA_COMPRESS_RATIO == 8
#        define GQA_BLOCK_READ intel_sub_group_2d_block_read_16b_8r16x2c
#    else
#        define GQA_BLOCK_READ intel_sub_group_2d_block_read_16b_16r16x2c
#    endif
#else
#    define GQA_2D 0
#endif

// Quantized caches read with 8-bit 2D messages: needs 64-byte aligned rows and a worker slice of whole 32-byte chunks.
#if GQA_2D && QSA_K_QUANT && (QSA_K_ROW_BYTES % 64) == 0 && ((QSA_XDH * QSA_K_BITS) % 256) == 0
#    define GQA_KQ2D 1
#    define GQA_KE (16 / QSA_K_BITS)  // channels per lane of a 32-byte chunk
#else
#    define GQA_KQ2D 0
#    define GQA_KE 1
#endif
#if GQA_2D && QSA_V_QUANT && (QSA_V_ROW_BYTES % 64) == 0 && ((QSA_XDV * QSA_V_BITS) % 256) == 0
#    define GQA_VQ2D 1
#    define GQA_VE (16 / QSA_V_BITS)
#else
#    define GQA_VQ2D 0
#    define GQA_VE 1
#endif
#if QSA_DENSE_QUANT && defined(QSA_K_SIGNED) && QSA_K_SIGNED
#    define GQA_K_SGN 1
#else
#    define GQA_K_SGN 0
#endif
#if QSA_DENSE_QUANT && defined(QSA_V_SIGNED) && QSA_V_SIGNED
#    define GQA_V_SGN 1
#else
#    define GQA_V_SGN 0
#endif

#if GQA_KQ2D || GQA_VQ2D
#    define GQA_KNC (QSA_XKT / GQA_KE)  // 32-byte chunks of a worker slice
#    define GQA_VNC (GQA_VT / GQA_VE)
#    define GQA_QNC (GQA_KNC > GQA_VNC ? GQA_KNC : GQA_VNC)
#    define GQA_QNP ((GQA_QNC + 1) / 2)  // chunk pairs (one 2-chunk message each)
#    if QSA_COMPRESS_RATIO == 2
#        define GQA_READ8_1C intel_sub_group_2d_block_read_8b_2r32x1c
#        define GQA_READ8_2C intel_sub_group_2d_block_read_8b_2r32x2c
#    elif QSA_COMPRESS_RATIO == 4
#        define GQA_READ8_1C intel_sub_group_2d_block_read_8b_4r32x1c
#        define GQA_READ8_2C intel_sub_group_2d_block_read_8b_4r32x2c
#    elif QSA_COMPRESS_RATIO == 8
#        define GQA_READ8_1C intel_sub_group_2d_block_read_8b_8r32x1c
#        define GQA_READ8_2C intel_sub_group_2d_block_read_8b_8r32x2c
#    else
#        define GQA_READ8_1C intel_sub_group_2d_block_read_8b_16r32x1c
#        define GQA_READ8_2C intel_sub_group_2d_block_read_8b_16r32x2c
#    endif

// One f16 row of a by-channel group record (scales or zero points) for the worker slice at `p`, with one subgroup block
// read (uniform address): out[c] (.s01 for E = 2) holds the E halves of channels c * 16 E + E lane .. + E - 1.
inline void FUNC(gqa_rec_row)(const __global uchar* p, const uint nc, const uint E, __private half4* out) {
    if (E == 2) {
        if (nc == 1) {
            out[0] = (half4)(as_half2(intel_sub_group_block_read((const __global uint*)p)), 0.0h, 0.0h);
        } else if (nc == 2) {
            const uint2 d = intel_sub_group_block_read2((const __global uint*)p);
            out[0] = (half4)(as_half2(d.s0), 0.0h, 0.0h);
            out[1] = (half4)(as_half2(d.s1), 0.0h, 0.0h);
        } else {
            unroll_for (uint c = 0; c < nc; c += 4) {
                const uint4 d = intel_sub_group_block_read4((const __global uint*)p + c * 16);
                out[c + 0] = (half4)(as_half2(d.s0), 0.0h, 0.0h);
                out[c + 1] = (half4)(as_half2(d.s1), 0.0h, 0.0h);
                out[c + 2] = (half4)(as_half2(d.s2), 0.0h, 0.0h);
                out[c + 3] = (half4)(as_half2(d.s3), 0.0h, 0.0h);
            }
        }
    } else {
#    ifdef cl_intel_subgroups_long
        if (nc == 1) {
            out[0] = as_half4(intel_sub_group_block_read_ul((const __global ulong*)p));
        } else {
            unroll_for (uint c = 0; c < nc; c += 2) {
                const ulong2 d = intel_sub_group_block_read_ul2((const __global ulong*)p + c * 16);
                out[c + 0] = as_half4(d.s0);
                out[c + 1] = as_half4(d.s1);
            }
        }
#    else
        unroll_for (uint c = 0; c < nc; c++) {
            out[c] = vload4(c * 16 + get_sub_group_local_id(), (const __global half*)p);
        }
#    endif
    }
}

// Issues the reads of a quantized step (worker slice from channel x0, nc chunks of E channels per lane): one message per
// block and chunk pair, code[j][p][h * R + r] = bytes (2 lane, 2 lane + 1) of chunk 2 p + h of row r of block j (rows
// past a short block read as zero, an empty block reads a valid row). By-channel caches (`by_channel`, paged keys) also
// get the f16 scales / zero points of the lane's channels per block, from the group record at `par` (zero for empty
// blocks); per-row caches take this lane's row parameters from the caller.
inline void FUNC(gqa_q2d_read)(const __global uchar* cache,
                               const uint off,
                               const uint par,
                               const uint valid_kv,
                               const uint x0,
                               const uint nc,
                               const uint E,
                               const int row_bytes,
                               const bool by_channel,
                               const uint head,
                               const uint sym,
                               const uint bits,
                               __private ushort (*code)[GQA_QNP][2 * GQA_R],
                               __private half4 (*cs)[GQA_QNC],
                               __private half4 (*cz)[GQA_QNC]) {
    const int xb = (int)(x0 * 2 / E);  // byte column of the slice
    unroll_for (uint j = 0; j < GQA_BLOCKS; j++) {
        const uint rows = valid_kv > j * GQA_R ? min(valid_kv - j * GQA_R, (uint)GQA_R) : 0u;
        const __global uchar* base = cache + sub_group_broadcast(off, j * GQA_R);
        unroll_for (uint p = 0; p < (nc + 1) / 2; p++) {
            if (2 * p + 1 < nc) {
                GQA_READ8_2C(base, row_bytes, (int)max(rows, 1u), row_bytes, (int2)(xb + (int)p * 64, 0), code[j][p]);
            } else {
                GQA_READ8_1C(base, row_bytes, (int)max(rows, 1u), row_bytes, (int2)(xb + (int)p * 64, 0), code[j][p]);
            }
        }
    }
    if (by_channel) {
        // A record (one group of a page) is as large as the codes of a block, so blocks of the same group as the
        // previous block reuse its parameters; empty blocks load nothing.
        const half zs = convert_half(FUNC_CALL(qsa_q_zero_sym)(bits));
        unroll_for (uint j = 0; j < GQA_BLOCKS; j++) {
            const uint pj = sub_group_broadcast(par, j * GQA_R);
            if (valid_kv <= j * GQA_R) {
                unroll_for (uint c = 0; c < nc; c++) {
                    cs[j][c] = (half4)((half)0.0f);
                    cz[j][c] = (half4)((half)0.0f);
                }
            } else if (j > 0 && pj == sub_group_broadcast(par, (j - 1) * GQA_R)) {
                unroll_for (uint c = 0; c < nc; c++) {
                    cs[j][c] = cs[j - 1][c];
                    cz[j][c] = cz[j - 1][c];
                }
            } else {
                const __global uchar* rec = cache + pj + x0 * 2;
                FUNC_CALL(gqa_rec_row)(rec, nc, E, cs[j]);
                if (sym) {
                    unroll_for (uint c = 0; c < nc; c++) {
                        cz[j][c] = (half4)zs;
                    }
                } else {
                    FUNC_CALL(gqa_rec_row)(rec + head * 2, nc, E, cz[j]);
                }
            }
        }
    }
}

// Dequantized tiles of a step read by gqa_q2d_read: x[c * E + q][r] = channel x0 + c * 16 E + E * lane + q of row r, in
// the f16 arithmetic of qsa_q_dq ((code - zp) * scale, each step rounded to half). Per-row caches use this lane's
// row parameters rs / rz (zero for rows past valid_kv, which zeroes them).
inline void FUNC(gqa_q2d_dequant)(__private const ushort (*code)[GQA_QNP][2 * GQA_R],
                                  __private const half4 (*cs)[GQA_QNC],
                                  __private const half4 (*cz)[GQA_QNC],
                                  const float rs,
                                  const float rz,
                                  const uint nc,
                                  const uint E,
                                  const bool by_channel,
                                  const bool sgn,
                                  __private half (*x)[16]) {
    unroll_for (uint r = 0; r < 16; r++) {
        const half s_row = convert_half(sub_group_broadcast(rs, r));
        const half z_row = convert_half(sub_group_broadcast(rz, r));
        const uint j = r / GQA_R;
        unroll_for (uint c = 0; c < nc; c++) {
            const ushort raw = code[j][c / 2][(c % 2) * GQA_R + r % GQA_R];
            unroll_for (uint q = 0; q < E; q++) {
                // The code as half without an int -> half conversion: 0x6400 | u is 1024 + u (u < 1024) exactly, and
                // subtracting 1024 (1152 for signed codes biased by 128) is exact too.
                ushort u;
                if (E == 2) {
                    u = q ? (ushort)(raw >> 8) : (ushort)(raw & 0xFF);
                    u = sgn ? (ushort)(u ^ 0x80) : u;
                } else {
                    u = (ushort)((raw >> (4 * q)) & 0xF);
                }
                const half v = as_half((ushort)(u | 0x6400)) - (sgn ? (half)1152.0f : (half)1024.0f);
                const half s = by_channel ? cs[j][c][q] : s_row;
                const half z = by_channel ? cz[j][c][q] : z_row;
                x[c * E + q][r] = (v - z) * s;
            }
        }
    }
}

// This lane's row parameters (lane = KV row) of a per-row quantized cache: paged by-token dword at byte offset `par`,
// or dense scale / zero point tensors at token index `par`. Zero for rows past valid_kv and for by-channel caches.
inline void FUNC(gqa_row_params)(const __global uchar* cache,
                                 const __global half* scale,
                                 const __global uchar* zp,
                                 const uint par,
                                 const uint valid_kv,
                                 const bool dense,
                                 const uint mode,
                                 const uint bits,
                                 const uint sym,
                                 const uint is_signed,
                                 const uint zp_f16,
                                 __private float* rs,
                                 __private float* rz) {
    float s = 0.0f, z = 0.0f;
    if (dense) {
        s = convert_float(scale[par]);
        const float zs = is_signed ? 0.0f : FUNC_CALL(qsa_q_zero_sym)(bits);
        z = sym ? zs : (zp_f16 ? convert_float(((const __global half*)zp)[par]) : convert_float(((const __global char*)zp)[par]));
    } else if (mode == 1) {
        const uint pd = *(const __global uint*)(cache + par);
        s = convert_float(as_half((ushort)(pd & 0xFFFFu)));
        z = sym ? FUNC_CALL(qsa_q_zero_sym)(bits) : convert_float(as_half((ushort)(pd >> 16)));
    }
    const bool valid = get_sub_group_local_id() < valid_kv;
    *rs = valid ? s : 0.0f;
    *rz = valid ? z : 0.0f;
}

#    if QSA_DENSE_QUANT
#        define GQA_K_PARAM_PTRS key_scale, QSA_KZP_PTR
#        define GQA_V_PARAM_PTRS value_scale, QSA_VZP_PTR
#        define GQA_K_ZPF16 QSA_K_ZP_F16
#        define GQA_V_ZPF16 QSA_V_ZP_F16
#    else
#        define GQA_K_PARAM_PTRS (const __global half*)key_cache, (const __global uchar*)key_cache
#        define GQA_V_PARAM_PTRS (const __global half*)value_cache, (const __global uchar*)value_cache
#        define GQA_K_ZPF16 0
#        define GQA_V_ZPF16 0
#    endif
#endif

#define GQA_S8(t, o) as_short8((ushort8)((t)[(o) + 0], (t)[(o) + 1], (t)[(o) + 2], (t)[(o) + 3], (t)[(o) + 4], (t)[(o) + 5], (t)[(o) + 6], (t)[(o) + 7]))
#define GQA_I8(t, o) as_int8((uint8)((t)[(o) + 0], (t)[(o) + 1], (t)[(o) + 2], (t)[(o) + 3], (t)[(o) + 4], (t)[(o) + 5], (t)[(o) + 6], (t)[(o) + 7]))

// K of one step as A operands: ka[k][h] holds rows 8h..8h+7 (element) x columns worker * XDH + 16 k + lane.
// Quantized K: `koff` is the byte offset of this lane's row and `kpar` its parameter locator (see QSA_LOAD_K_COLS).
inline void FUNC(gqa_load_k)(QSA_QP_DECL const __global QSA_KC_T* key_cache,
                             const uint koff,
                             const uint kpar,
                             const uint valid_kv,
                             const bool contiguous,
                             const uint worker,
                             __private short8 (*ka)[2]) {
    const int row_bytes = QSA_K_HEAD_SIZE * 2;
    const uint x0 = worker * QSA_XDH;
#if QSA_K_QUANT
    {
        unroll_for (uint i = 0; i < 16; i++) {
            const uint ro = sub_group_broadcast(koff, i);
            const uint po = sub_group_broadcast(kpar, i);
            float x[QSA_XKT];
            unroll_for (uint k = 0; k < QSA_XKT; k++) {
                x[k] = 0.0f;
            }
            if (i < valid_kv) {
                QSA_LOAD_K_COLS(ro, po, x0, x, QSA_XKT);
            }
            unroll_for (uint k = 0; k < QSA_XKT; k++) {
                ka[k][i / 8][i % 8] = as_short(convert_half_rte(x[k]));
            }
        }
    }
#elif GQA_2D
    {
        if (contiguous) {
            const __global half* base = key_cache + sub_group_broadcast(koff, 0);
            unroll_for (uint m = 0; m < QSA_XKT / 2; m++) {
                ushort t[32];
                intel_sub_group_2d_block_read_16b_16r16x2c(base, row_bytes, 16, row_bytes, (int2)(x0 + m * 32, 0), t);
                ka[2 * m][0] = GQA_S8(t, 0);
                ka[2 * m][1] = GQA_S8(t, 8);
                ka[2 * m + 1][0] = GQA_S8(t, 16);
                ka[2 * m + 1][1] = GQA_S8(t, 24);
            }
            return;
        }
        // Every read is issued before any is consumed (an empty block reads one row of a valid address and is
        // masked), so the latencies overlap instead of serializing on per-block branches.
        ushort t[GQA_BLOCKS][QSA_XKT / 2][2 * GQA_R];
        unroll_for (uint j = 0; j < GQA_BLOCKS; j++) {
            const uint rows = valid_kv > j * GQA_R ? min(valid_kv - j * GQA_R, (uint)GQA_R) : 0u;
            const __global half* base = key_cache + sub_group_broadcast(koff, j * GQA_R);
            unroll_for (uint m = 0; m < QSA_XKT / 2; m++) {
                GQA_BLOCK_READ(base, row_bytes, (int)max(rows, 1u), row_bytes, (int2)(x0 + m * 32, 0), t[j][m]);
            }
        }
        unroll_for (uint j = 0; j < GQA_BLOCKS; j++) {
            const bool empty = valid_kv <= j * GQA_R;
            unroll_for (uint m = 0; m < QSA_XKT / 2; m++) {
                unroll_for (uint r = 0; r < GQA_R; r++) {
                    ka[2 * m][(j * GQA_R + r) / 8][(j * GQA_R + r) % 8] = empty ? 0 : (short)t[j][m][r];
                    ka[2 * m + 1][(j * GQA_R + r) / 8][(j * GQA_R + r) % 8] = empty ? 0 : (short)t[j][m][GQA_R + r];
                }
            }
        }
    }
#else
    unroll_for (uint i = 0; i < 16; i++) {
        ushort r[QSA_XKT];
        const uint ko = sub_group_broadcast(koff, i);
        if (i < valid_kv) {
            FUNC_CALL(qsa_rd)((const __global ushort*)(key_cache + ko + x0), r, QSA_XKT);
        } else {
            unroll_for (uint k = 0; k < QSA_XKT; k++) {
                r[k] = 0;
            }
        }
        unroll_for (uint k = 0; k < QSA_XKT; k++) {
            ka[k][i / 8][i % 8] = (short)r[k];
        }
    }
#endif
}

// V of one step as VNNI B operands: vb[t] element j packs rows 2j, 2j+1 of column worker * XDV + 16 t + lane.
inline void FUNC(gqa_load_v)(QSA_QP_DECL const __global QSA_VC_T* value_cache,
                             const uint voff,
                             const uint vpar,
                             const uint valid_kv,
                             const bool contiguous,
                             const uint worker,
                             __private int8* vb) {
    const int row_bytes = QSA_V_HEAD_SIZE * 2;
    const uint x0 = worker * QSA_XDV;
#if QSA_V_QUANT
    {
        unroll_for (uint i = 0; i < 16; i += 2) {
            const uint ro0 = sub_group_broadcast(voff, i);
            const uint ro1 = sub_group_broadcast(voff, i + 1);
            const uint po0 = sub_group_broadcast(vpar, i);
            const uint po1 = sub_group_broadcast(vpar, i + 1);
            float v0[GQA_VT], v1[GQA_VT];
            unroll_for (uint t = 0; t < GQA_VT; t++) {
                v0[t] = 0.0f;
                v1[t] = 0.0f;
            }
            if (i < valid_kv) {
                QSA_LOAD_V_COLS(ro0, po0, x0, v0, GQA_VT);
            }
            if (i + 1 < valid_kv) {
                QSA_LOAD_V_COLS(ro1, po1, x0, v1, GQA_VT);
            }
            unroll_for (uint t = 0; t < GQA_VT; t++) {
                vb[t][i / 2] = (int)((uint)as_ushort(convert_half_rte(v0[t])) | ((uint)as_ushort(convert_half_rte(v1[t])) << 16));
            }
        }
    }
#elif GQA_2D
    {
        if (contiguous) {
            const __global half* base = value_cache + sub_group_broadcast(voff, 0);
            unroll_for (uint m = 0; m < GQA_VT / 2; m++) {
                uint t[16];
                intel_sub_group_2d_block_read_transform_16b_16r16x2c(base, row_bytes, 16, row_bytes, (int2)(x0 + m * 32, 0), t);
                vb[2 * m] = GQA_I8(t, 0);
                vb[2 * m + 1] = GQA_I8(t, 8);
            }
            return;
        }
        // A VNNI transform read is at least 16 rows high, so short blocks are read plainly (all reads first, as for K)
        // and packed in registers.
        ushort t[GQA_BLOCKS][GQA_VT / 2][2 * GQA_R];
        unroll_for (uint j = 0; j < GQA_BLOCKS; j++) {
            const uint rows = valid_kv > j * GQA_R ? min(valid_kv - j * GQA_R, (uint)GQA_R) : 0u;
            const __global half* base = value_cache + sub_group_broadcast(voff, j * GQA_R);
            unroll_for (uint m = 0; m < GQA_VT / 2; m++) {
                GQA_BLOCK_READ(base, row_bytes, (int)max(rows, 1u), row_bytes, (int2)(x0 + m * 32, 0), t[j][m]);
            }
        }
        unroll_for (uint j = 0; j < GQA_BLOCKS; j++) {
            const bool empty = valid_kv <= j * GQA_R;
            unroll_for (uint m = 0; m < GQA_VT / 2; m++) {
                unroll_for (uint q = 0; q < GQA_R / 2; q++) {
                    const uint lo = (uint)t[j][m][2 * q] | ((uint)t[j][m][2 * q + 1] << 16);
                    const uint hi = (uint)t[j][m][GQA_R + 2 * q] | ((uint)t[j][m][GQA_R + 2 * q + 1] << 16);
                    vb[2 * m][j * GQA_R / 2 + q] = empty ? 0 : (int)lo;
                    vb[2 * m + 1][j * GQA_R / 2 + q] = empty ? 0 : (int)hi;
                }
            }
        }
    }
#else
    unroll_for (uint i = 0; i < 16; i += 2) {
        ushort r0[GQA_VT], r1[GQA_VT];
        const uint vo0 = sub_group_broadcast(voff, i);
        const uint vo1 = sub_group_broadcast(voff, i + 1);
        unroll_for (uint t = 0; t < GQA_VT; t++) {
            r0[t] = 0;
            r1[t] = 0;
        }
        if (i < valid_kv) {
            FUNC_CALL(qsa_rd)((const __global ushort*)(value_cache + vo0 + x0), r0, GQA_VT);
        }
        if (i + 1 < valid_kv) {
            FUNC_CALL(qsa_rd)((const __global ushort*)(value_cache + vo1 + x0), r1, GQA_VT);
        }
        unroll_for (uint t = 0; t < GQA_VT; t++) {
            vb[t][i / 2] = (int)((uint)r0[t] | ((uint)r1[t] << 16));
        }
    }
#endif
}

// One KV step of the online softmax for this worker: QK partials summed over the workers through `slm_st`, softmax
// with a lazy rescale, P^T through the worker's `slm_p` tile, then PV into ro.
inline void FUNC(gqa_step)(QSA_QP_DECL const __global QSA_KC_T* key_cache,
                           const __global QSA_VC_T* value_cache,
                           const uint koff,
                           const uint voff,
                           const uint kpar,
                           const uint vpar,
                           const uint valid_kv,
                           const bool kcont,
                           const bool vcont,
                           const uint worker,
                           __private const int8* rq,
                           __local float* slm_st,  // [W][2][8][SG] of the current slot
                           __local uint8* slm_p,   // [16] of this worker and slot
                           __private float* running_max,
                           __private float* running_sum,
                           __private float8 (*ro)[GQA_VT]) {
    const uint lane = get_sub_group_local_id();
    short8 ka[QSA_XKT][2];
#if GQA_KQ2D
    ushort kcode[GQA_BLOCKS][GQA_QNP][2 * GQA_R];
    half4 kcs[GQA_BLOCKS][GQA_QNC], kcz[GQA_BLOCKS][GQA_QNC];
    float krs, krz;
    FUNC_CALL(gqa_row_params)((const __global uchar*)key_cache, GQA_K_PARAM_PTRS, kpar, valid_kv, QSA_DENSE_QUANT, QSA_K_MODE, QSA_K_BITS, QSA_K_SYM, GQA_K_SGN,
                              GQA_K_ZPF16, &krs, &krz);
    FUNC_CALL(gqa_q2d_read)((const __global uchar*)key_cache, koff, kpar, valid_kv, worker * QSA_XDH, GQA_KNC, GQA_KE, QSA_K_ROW_BYTES, QSA_K_MODE == 2,
                            QSA_K_HEAD_SIZE, QSA_K_SYM, QSA_K_BITS, kcode, kcs, kcz);
#else
    FUNC_CALL(gqa_load_k)(QSA_QP_ARGS key_cache, koff, kpar, valid_kv, kcont, worker, ka);
#endif
    // V is in flight during QK and the softmax.
    int8 vb[GQA_VT];
#if GQA_VQ2D
    ushort vcode[GQA_BLOCKS][GQA_QNP][2 * GQA_R];
    half4 vcs[GQA_BLOCKS][GQA_QNC], vcz[GQA_BLOCKS][GQA_QNC];
    float vrs, vrz;
    FUNC_CALL(gqa_row_params)((const __global uchar*)value_cache, GQA_V_PARAM_PTRS, vpar, valid_kv, QSA_DENSE_QUANT, QSA_V_MODE, QSA_V_BITS, QSA_V_SYM, GQA_V_SGN,
                              GQA_V_ZPF16, &vrs, &vrz);
    FUNC_CALL(gqa_q2d_read)((const __global uchar*)value_cache, voff, vpar, valid_kv, worker * QSA_XDV, GQA_VNC, GQA_VE, QSA_V_ROW_BYTES, false,
                            QSA_V_HEAD_SIZE, QSA_V_SYM, QSA_V_BITS, vcode, vcs, vcz);
#else
    FUNC_CALL(gqa_load_v)(QSA_QP_ARGS value_cache, voff, vpar, valid_kv, vcont, worker, vb);
#endif
#if GQA_KQ2D
    {
        half x[QSA_XKT][16];
        FUNC_CALL(gqa_q2d_dequant)(kcode, kcs, kcz, krs, krz, GQA_KNC, GQA_KE, QSA_K_MODE == 2, GQA_K_SGN, x);
        unroll_for (uint k = 0; k < QSA_XKT; k++) {
            unroll_for (uint i = 0; i < 16; i++) {
                ka[k][i / 8][i % 8] = as_short(x[k][i]);
            }
        }
    }
#endif

    // QK partial of this worker, summed over the workers in the fixed order 0..W-1 (bit-identical everywhere).
    float8 st0 = (float8)(0.0f);
    float8 st1 = (float8)(0.0f);
    unroll_for (uint k = 0; k < QSA_XKT; k++) {
        st0 = intel_sub_group_f16_f16_matrix_mad_k16(ka[k][0], rq[k], st0);
        st1 = intel_sub_group_f16_f16_matrix_mad_k16(ka[k][1], rq[k], st1);
    }
    QSA_SLM_WRITE8(slm_st + worker * QSA_XST_FLOATS, st0);
    QSA_SLM_WRITE8(slm_st + worker * QSA_XST_FLOATS + 8 * QSA_SG, st1);
    barrier(CLK_LOCAL_MEM_FENCE);
    st0 = QSA_SLM_READ8(slm_st);
    st1 = QSA_SLM_READ8(slm_st + 8 * QSA_SG);
    unroll_for (uint w = 1; w < QSA_XW; w++) {
        st0 += QSA_SLM_READ8(slm_st + w * QSA_XST_FLOATS);
        st1 += QSA_SLM_READ8(slm_st + w * QSA_XST_FLOATS + 8 * QSA_SG);
    }
    float st[16];
    vstore8(st0, 0, st);
    vstore8(st1, 0, st + 8);

    // Padded head lanes see zero Q (finite scores, never stored): only the rows past valid_kv need masking.
    if (valid_kv != 16) {
        unroll_for (uint i = 0; i < 16; i++) {
            st[i] = i >= valid_kv ? QSA_NEG_INF : st[i];
        }
    }
    float tile_max = st[0];
    unroll_for (uint i = 1; i < 16; i++) {
        tile_max = fmax(tile_max, st[i]);
    }
    // Lazy rescale: the running max only moves when some lane grows by more than 2^GQA_RESCALE_LOG2.
    float m = *running_max;
    float s = *running_sum;
    if (sub_group_any(tile_max > m + GQA_RESCALE_LOG2)) {
        const float new_max = fmax(m, tile_max);
        const float corr = native_exp2(m - new_max);
        s *= corr;
        m = new_max;
        unroll_for (uint hh = 0; hh < GQA_HH; hh++) {
            float8 c;
            unroll_for (uint i = 0; i < 8; i++) {
                c[i] = sub_group_broadcast(corr, hh * 8 + i);
            }
            unroll_for (uint t = 0; t < GQA_VT; t++) {
                ro[hh][t] *= c;
            }
        }
    }
    float p[16];
    unroll_for (uint i = 0; i < 16; i++) {
        p[i] = native_exp2(st[i] - m);
        s += p[i];
    }
    *running_max = m;
    *running_sum = s;

    // P^T through SLM: lane h stores its row P[h][0..15]; a block read then gives lane = KV row, element = head.
    slm_p[lane] = (uint8)(as_uint(convert_half2_rte((float2)(p[0], p[1]))), as_uint(convert_half2_rte((float2)(p[2], p[3]))),
                          as_uint(convert_half2_rte((float2)(p[4], p[5]))), as_uint(convert_half2_rte((float2)(p[6], p[7]))),
                          as_uint(convert_half2_rte((float2)(p[8], p[9]))), as_uint(convert_half2_rte((float2)(p[10], p[11]))),
                          as_uint(convert_half2_rte((float2)(p[12], p[13]))), as_uint(convert_half2_rte((float2)(p[14], p[15]))));
    sub_group_barrier(CLK_LOCAL_MEM_FENCE);
#if GQA_VQ2D
    {
        half x[GQA_VT][16];
        FUNC_CALL(gqa_q2d_dequant)(vcode, vcs, vcz, vrs, vrz, GQA_VNC, GQA_VE, false, GQA_V_SGN, x);
        unroll_for (uint t = 0; t < GQA_VT; t++) {
            unroll_for (uint j = 0; j < 8; j++) {
                vb[t][j] = (int)((uint)as_ushort(x[t][2 * j]) | ((uint)as_ushort(x[t][2 * j + 1]) << 16));
            }
        }
    }
#endif
    unroll_for (uint hh = 0; hh < GQA_HH; hh++) {
        const __local ushort* pt = (const __local ushort*)slm_p + hh * 128;
#ifdef cl_intel_subgroup_local_block_io
        const short8 pa = as_short8(intel_sub_group_block_read_us8(pt));
#else
        const short8 pa = (short8)((short)pt[lane], (short)pt[16 + lane], (short)pt[32 + lane], (short)pt[48 + lane],
                                   (short)pt[64 + lane], (short)pt[80 + lane], (short)pt[96 + lane], (short)pt[112 + lane]);
#endif
        unroll_for (uint t = 0; t < GQA_VT; t++) {
            ro[hh][t] = intel_sub_group_f16_f16_matrix_mad_k16(pa, vb[t], ro[hh][t]);
        }
    }
}

REQD_SUB_GROUP_SIZE(SUBGROUP_SIZE)
KERNEL(sparse_attention_gqa_dpas)(QSA_ATT_INPUT_ARGS,
#if QSA_GQA_SPLIT
                                  __global float* partial_acc,  // [tokens, Hq, P, Dv]
                                  __global float* partial_ml,   // [tokens, Hq, P, 2]
#else
                                  __global half* output,        // paged: [tokens, Hq*Dv]; dense: [B, Hq, L, Dv]
#endif
#if QSA_ADDRESS_MODE == 0
                                  const uint num_entries,
                                  const uint num_seqs,
                                  const uint q_base,
                                  const uint q_pitch
#else
                                  const uint num_entries,
                                  const uint seq_len,
                                  const uint history_len,
                                  const uint k_pitch,
                                  const uint v_pitch
#    if QSA_DENSE_QUANT
                                  ,
                                  const uint k_ppitch,
                                  const uint v_ppitch
#    endif
#endif
) {
    const uint HQ = QSA_HEADS;
    const uint HKV = QSA_KV_HEADS;
    const uint DH = QSA_K_HEAD_SIZE;
    const uint DV = QSA_V_HEAD_SIZE;
    const uint R = QSA_COMPRESS_RATIO;
    const uint K_TOPK = QSA_BLOCK_TOPK;
    const uint HEADS = GQA_HEADS;

    __local float slm_st[2][QSA_XW][2][8][QSA_SG];
    __local uint8 slm_p[2][QSA_XW][16];  // per worker: row h = P[head h][kv rows 0..15] as halves

    const uint worker = get_sub_group_id();
    const uint lane = get_sub_group_local_id();
    const uint entry = get_group_id(0);
    const uint kv_head = get_group_id(1);
    if (kv_head >= HKV || entry >= num_entries) {
        return;
    }
    const uint head_base = kv_head * HEADS;
    const uint lane_blk = lane / R;
    const uint lane_rr = lane % R;

#if QSA_ADDRESS_MODE == 0
    const uint token = entry;
    uint seq = 0, abs_pos = 0;
    if (!FUNC_CALL(qsa_resolve_token_paged)(subsequence_begins, past_lens, num_seqs, token, &seq, &abs_pos)) {
        return;
    }
    const uint page_begin = (uint)block_indices_begins[seq];
    const uint count = (uint)selected_counts[token];
    const uint sel_base = token * K_TOPK;
    const uint tail_begin = (abs_pos + 1) / R * R;
    const uint tail_rows = abs_pos + 1 - tail_begin;
    const uint q_head0 = q_base + token * q_pitch + head_base * DH;
#    define GQA_Q_HEAD(h) (q_head0 + (h) * DH)
#    define GQA_OUT_ROW(h) (token * HQ + head_base + (h))
#else
    const uint batch = entry / seq_len;
    const uint l = entry - batch * seq_len;
    uint kl = (uint)key_length[batch];
    kl = kl < history_len ? kl : history_len;
    const uint shift = history_len - kl;
    const bool pad_row = kl + l < seq_len;
    const uint abs_pos = kl + l - seq_len;
    const uint count = pad_row ? 0 : (uint)selected_counts[entry];
    const uint sel_base = entry * K_TOPK;
    const uint tail_begin = pad_row ? 0 : (abs_pos + 1) / R * R;
    const uint tail_rows = pad_row ? 0 : abs_pos + 1 - tail_begin;
#    define GQA_Q_HEAD(h) ((((batch * HQ) + head_base + (h)) * seq_len + l) * DH)
#    define GQA_OUT_ROW(h) (((batch * HQ) + head_base + (h)) * seq_len + l)
#endif

    // Q as the B operand: this lane's own head row slice, pre-scaled by scale * log2(e) in the half domain.
    // Tile k = c E + q holds channels c * 16 E + E i + q (i = element), the K channel order of the step tiles.
    int8 rq[QSA_XKT];
    unroll_for (uint c = 0; c < QSA_XKT / GQA_KE; c++) {
        half qc[16 * GQA_KE];
        unroll_for (uint u = 0; u < GQA_KE; u++) {
            half16 v = (half16)((half)0.0f);
            if (lane < HEADS) {
                v = vload16(0, query + GQA_Q_HEAD(lane) + worker * QSA_XDH + (c * GQA_KE + u) * 16);
            }
            unroll_for (uint i = 0; i < 16; i++) {
                qc[16 * u + i] = v[i];
            }
        }
        unroll_for (uint q = 0; q < GQA_KE; q++) {
            half16 qh;
            unroll_for (uint i = 0; i < 16; i++) {
                qh[i] = qc[GQA_KE * i + q];
            }
            qh = qh * (half)((float)QSA_SCALE * QSA_LOG2E);
            rq[c * GQA_KE + q] = as_int8(qh);
        }
    }

    float running_max = QSA_NEG_INF;  // of head `lane`
    float running_sum = 0.0f;
    float8 ro[GQA_HH][GQA_VT];  // element = head 8 hh + i, lane = output column 16 t + lane of the worker slice
    unroll_for (uint hh = 0; hh < GQA_HH; hh++) {
        unroll_for (uint t = 0; t < GQA_VT; t++) {
            ro[hh][t] = (float8)(0.0f);
        }
    }

    const uint steps = (count + GQA_BLOCKS - 1) / GQA_BLOCKS;
#if QSA_GQA_SPLIT
    const uint P = QSA_SPLIT_PARTITIONS;
    const uint part = get_group_id(2);
    // The last partition exclusively owns the tail step.
    const uint step_begin = steps * part / P;
    const uint step_end = steps * (part + 1) / P + (part == P - 1 && tail_rows != 0 ? 1 : 0);
#else
    const uint step_begin = 0;
    const uint step_end = steps + (tail_rows != 0 ? 1 : 0);
#endif

#define GQA_LOAD_ID(step) (((step) < steps && (step) * GQA_BLOCKS + lane_blk < count) ? selected_blocks[sel_base + (step) * GQA_BLOCKS + lane_blk] : -1)
// Lane = KV row of the step: its validity and absolute position (step == steps is the tail).
#define GQA_META(step, id, lv, pos)          \
    if ((step) == steps) {                   \
        lv = lane < tail_rows;               \
        pos = tail_begin + lane;             \
    } else {                                 \
        lv = (id) >= 0;                      \
        pos = (uint)(id) * R + lane_rr;      \
    }
#if QSA_ADDRESS_MODE == 0
#    define GQA_PAGE(lv, pos) ((lv) ? (uint)block_indices[page_begin + (pos) / QSA_PAGE_SIZE] : 0u)
#else
#    define GQA_PAGE(lv, pos) 0u
#endif

    // Software-pipelined metadata: the next step's pages and the following step's ids are in flight while a step
    // computes, so the KV loads never wait on the id -> page chain.
    bool cur_lv;
    uint cur_pos;
    const int first_id = GQA_LOAD_ID(step_begin);
    GQA_META(step_begin, first_id, cur_lv, cur_pos);
    uint cur_page = GQA_PAGE(cur_lv, cur_pos);
    int next_id = GQA_LOAD_ID(step_begin + 1);

    for (uint step = step_begin; step < step_end; step++) {
        bool nlv;
        uint npos;
        GQA_META(step + 1, next_id, nlv, npos);
        const uint npage = GQA_PAGE(nlv, npos);
        next_id = GQA_LOAD_ID(step + 2);

        const bool lv = cur_lv;
        const uint pos = cur_pos;
        uint koff = 0, voff = 0, kpar = 0, vpar = 0;
        if (lv) {
#if QSA_ADDRESS_MODE == 0
#    if QSA_K_QUANT
            QSA_K_LOC(cur_page, pos % QSA_PAGE_SIZE, &koff, &kpar);
#    else
            koff = FUNC_CALL(qsa_kv_cache_offset)(cur_page, kv_head, pos % QSA_PAGE_SIZE, HKV, QSA_PAGE_SIZE, DH);
#    endif
#    if QSA_V_QUANT
            QSA_V_LOC(cur_page, pos % QSA_PAGE_SIZE, &voff, &vpar);
#    else
            voff = FUNC_CALL(qsa_kv_cache_offset)(cur_page, kv_head, pos % QSA_PAGE_SIZE, HKV, QSA_PAGE_SIZE, DV);
#    endif
#else
#    if QSA_K_QUANT
            koff = QSA_KROWB(pos);
            kpar = QSA_KPAR(pos);
#    else
            koff = QSA_KROW(pos);
#    endif
#    if QSA_V_QUANT
            voff = QSA_VROWB(pos);
            vpar = QSA_VPAR(pos);
#    else
            voff = QSA_VROW(pos);
#    endif
#endif
        }
        const uint valid_kv = sub_group_reduce_add(lv ? 1u : 0u);
        // A step read with one 2D message per operand needs ascending contiguous f16 rows.
        bool kcont = false, vcont = false;
#if !QSA_K_QUANT
        kcont = valid_kv == 16 && sub_group_all(koff == sub_group_broadcast(koff, 0) + lane * DH);
#endif
#if !QSA_V_QUANT
        vcont = valid_kv == 16 && sub_group_all(voff == sub_group_broadcast(voff, 0) + lane * DV);
#endif

        FUNC_CALL(gqa_step)(QSA_QP_ARGS key_cache, value_cache, koff, voff, kpar, vpar, valid_kv, kcont, vcont, worker, rq, &slm_st[step & 1][0][0][0][0],
                            &slm_p[step & 1][worker][0], &running_max, &running_sum, ro);
        cur_lv = nlv;
        cur_pos = npos;
        cur_page = npage;
    }

    // Never store the padded heads: that would overwrite the next KV group or token.
#if QSA_GQA_SPLIT
    if (worker == 0 && lane < HEADS) {
        // The reduce pass expects natural-log maxima; scores here are in log2 units.
        const uint prow = GQA_OUT_ROW(lane) * P + part;
        partial_ml[prow * 2] = running_max * 0.6931471805599453f;
        partial_ml[prow * 2 + 1] = running_sum;
    }
    unroll_for (uint h = 0; h < GQA_HEADS; h++) {
#    if GQA_VE == 1
        __global uint* dst = (__global uint*)(partial_acc + (GQA_OUT_ROW(h) * P + part) * DV + worker * QSA_XDV);
        unroll_for (uint t = 0; t < GQA_VT; t++) {
            intel_sub_group_block_write(dst + t * 16, as_uint(ro[h / 8][t][h % 8]));
        }
#    else
        // Tile c E + q holds column c * 16 E + E lane + q: each lane stores its E adjacent columns.
        __global float* dst = partial_acc + (GQA_OUT_ROW(h) * P + part) * DV + worker * QSA_XDV;
        unroll_for (uint c = 0; c < GQA_VNC; c++) {
#        if GQA_VE == 2
            vstore2((float2)(ro[h / 8][2 * c][h % 8], ro[h / 8][2 * c + 1][h % 8]), c * 16 + lane, dst);
#        else
            vstore4((float4)(ro[h / 8][4 * c][h % 8], ro[h / 8][4 * c + 1][h % 8], ro[h / 8][4 * c + 2][h % 8], ro[h / 8][4 * c + 3][h % 8]),
                    c * 16 + lane,
                    dst);
#        endif
        }
#    endif
    }
#else
    const float inv = running_sum > 0.0f ? 1.0f / running_sum : 0.0f;
    unroll_for (uint h = 0; h < GQA_HEADS; h++) {
        const float ih = sub_group_broadcast(inv, h);
#    if GQA_VE == 1
        __global ushort* dst = (__global ushort*)(output + GQA_OUT_ROW(h) * DV + worker * QSA_XDV);
        unroll_for (uint t = 0; t < GQA_VT; t++) {
            intel_sub_group_block_write_us(dst + t * 16, as_ushort(convert_half_rte(ro[h / 8][t][h % 8] * ih)));
        }
#    else
        __global half* dst = output + GQA_OUT_ROW(h) * DV + worker * QSA_XDV;
        unroll_for (uint c = 0; c < GQA_VNC; c++) {
#        if GQA_VE == 2
            vstore2(convert_half2_rte((float2)(ro[h / 8][2 * c][h % 8], ro[h / 8][2 * c + 1][h % 8]) * ih), c * 16 + lane, dst);
#        else
            vstore4(convert_half4_rte((float4)(ro[h / 8][4 * c][h % 8], ro[h / 8][4 * c + 1][h % 8], ro[h / 8][4 * c + 2][h % 8], ro[h / 8][4 * c + 3][h % 8]) * ih),
                    c * 16 + lane,
                    dst);
#        endif
        }
#    endif
    }
#endif
}
