// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "repack_gguf_moe_weights.hpp"

#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <vector>

#include "openvino/core/parallel.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "openvino/runtime/aligned_buffer.hpp"
#include "openvino/util/pp.hpp"
#include "ov_ops/moe_compressed.hpp"

namespace ov::intel_gpu {

namespace {

// ============================================================================================
// SG (sub-group block-read) repack for the Q4_K / Q5_K / Q6_K native GGUF MoE GEMV kernels
// (ocl_v2/moe_gguf_sg_gemv.cl). See moe_gguf_sg_gemv.cl and
// q4k_moe_gemv/test_moe_gemv_sg_kernels.py (pack_qk_pqs_sg / pack_qk_psl_sg / pack_q5k_pqh_sg /
// pack_q6k_pqh_sg / pack_q6k_ps_sg / pack_q6k_pd_sg) for the reference byte layout this mirrors.
//
// Mirror of moe_3gemm_swiglu_opt.cpp's gguf_moe_sg_enabled(): keep the gate identical so the
// transform and the impl agree on whether the weights were repacked. Default ON: the new "SG"
// kernels are the intended replacement for the (slower) raw-GGUF-block decode kernels for these
// three types, so packing runs unconditionally unless explicitly disabled.
// ============================================================================================
bool moe_sg_enabled() {
    if (const char* env = std::getenv("OV_GPU_GGUF_MOE_SG")) {
        return std::atol(env) != 0;
    }
    return true;
}

constexpr size_t kSgOPG = 16;

// ---- Stage A: raw GGUF block -> "shuffled" (row-major, separated-section) per-block layout ----
// Identical byte transform to shuffle_q4k / shuffle_q5k / shuffle_q6k in
// q4k_moe_gemv/test_moe_gemv_sg_kernels.py (verified byte-exact against the python reference).

// Shared by Q4_K/Q5_K: 12-byte packed 6-bit scale/min record (ggml get_scale_min_k4) -> 16-byte
// psl "SoA seed" record (sl[8]x4bit | ml[8]x4bit | sh[8]x2bit | mh[8]x2bit | d(f16) | dmin(f16),
// bit-packed 4 values/byte for sl/ml and 4 values/byte for sh/mh as in the reference).
inline void scale_min_to_psl(const uint8_t* d0, const uint8_t* d1, const uint8_t* sc, uint8_t* opsl) {
    const uint8_t sc0 = sc[0], sc1 = sc[1], sc2 = sc[2], sc3 = sc[3];
    const uint8_t sc4 = sc[4], sc5 = sc[5], sc6 = sc[6], sc7 = sc[7];
    const uint8_t sc8 = sc[8], sc9 = sc[9], sc10 = sc[10], sc11 = sc[11];

    uint8_t rs[8], rm[8];
    rs[0] = sc0 & 63;
    rs[1] = sc1 & 63;
    rs[2] = sc2 & 63;
    rs[3] = sc3 & 63;
    rs[4] = static_cast<uint8_t>((sc8 & 0x0F) | ((sc0 >> 2) & 0x30));
    rs[5] = static_cast<uint8_t>((sc9 & 0x0F) | ((sc1 >> 2) & 0x30));
    rs[6] = static_cast<uint8_t>((sc10 & 0x0F) | ((sc2 >> 2) & 0x30));
    rs[7] = static_cast<uint8_t>((sc11 & 0x0F) | ((sc3 >> 2) & 0x30));
    rm[0] = sc4 & 63;
    rm[1] = sc5 & 63;
    rm[2] = sc6 & 63;
    rm[3] = sc7 & 63;
    rm[4] = static_cast<uint8_t>(((sc8 >> 4) & 0x0F) | ((sc4 >> 2) & 0x30));
    rm[5] = static_cast<uint8_t>(((sc9 >> 4) & 0x0F) | ((sc5 >> 2) & 0x30));
    rm[6] = static_cast<uint8_t>(((sc10 >> 4) & 0x0F) | ((sc6 >> 2) & 0x30));
    rm[7] = static_cast<uint8_t>(((sc11 >> 4) & 0x0F) | ((sc7 >> 2) & 0x30));

    opsl[0] = static_cast<uint8_t>((rs[0] & 0x0F) | ((rs[1] & 0x0F) << 4));
    opsl[1] = static_cast<uint8_t>((rs[2] & 0x0F) | ((rs[3] & 0x0F) << 4));
    opsl[2] = static_cast<uint8_t>((rs[4] & 0x0F) | ((rs[5] & 0x0F) << 4));
    opsl[3] = static_cast<uint8_t>((rs[6] & 0x0F) | ((rs[7] & 0x0F) << 4));
    opsl[4] = static_cast<uint8_t>((rm[0] & 0x0F) | ((rm[1] & 0x0F) << 4));
    opsl[5] = static_cast<uint8_t>((rm[2] & 0x0F) | ((rm[3] & 0x0F) << 4));
    opsl[6] = static_cast<uint8_t>((rm[4] & 0x0F) | ((rm[5] & 0x0F) << 4));
    opsl[7] = static_cast<uint8_t>((rm[6] & 0x0F) | ((rm[7] & 0x0F) << 4));
    opsl[8] = static_cast<uint8_t>(((rs[0] & 0x30) >> 4) | ((rs[1] & 0x30) >> 2) | (rs[2] & 0x30) | ((rs[3] & 0x30) << 2));
    opsl[9] = static_cast<uint8_t>(((rs[4] & 0x30) >> 4) | ((rs[5] & 0x30) >> 2) | (rs[6] & 0x30) | ((rs[7] & 0x30) << 2));
    opsl[10] = static_cast<uint8_t>(((rm[0] & 0x30) >> 4) | ((rm[1] & 0x30) >> 2) | (rm[2] & 0x30) | ((rm[3] & 0x30) << 2));
    opsl[11] = static_cast<uint8_t>(((rm[4] & 0x30) >> 4) | ((rm[5] & 0x30) >> 2) | (rm[6] & 0x30) | ((rm[7] & 0x30) << 2));
    opsl[12] = d0[0];
    opsl[13] = d0[1];
    opsl[14] = d1[0];
    opsl[15] = d1[1];
}

// Shared by Q4_K/Q5_K: 4 chunks of 32-byte packed-nibble qs -> 128-byte opqs (8 sub-blocks x 16
// bytes, low/high nibble pairs already reorganized so a single mask extracts a whole sub-block).
inline void qs_to_opqs(const uint8_t* qs, uint8_t* opqs) {
    for (int gc = 0; gc < 4; ++gc) {
        const uint8_t* qc = qs + gc * 32;
        const int j0 = 2 * gc, j1 = 2 * gc + 1;
        for (int i = 0; i < 16; ++i) {
            const uint8_t lo_i = qc[i] & 0x0F;
            const uint8_t lo_i16 = qc[16 + i] & 0x0F;
            const uint8_t hi_i = (qc[i] >> 4) & 0x0F;
            const uint8_t hi_i16 = (qc[16 + i] >> 4) & 0x0F;
            opqs[j0 * 16 + i] = static_cast<uint8_t>(lo_i | (lo_i16 << 4));
            opqs[j1 * 16 + i] = static_cast<uint8_t>(hi_i | (hi_i16 << 4));
        }
    }
}

// Q4_K raw block (144B: d(2) dmin(2) sc(12) qs(128)) -> opqs(128B) + opsl(16B).
inline void shuffle_q4k_block(const uint8_t* blk, uint8_t* opqs, uint8_t* opsl) {
    scale_min_to_psl(blk + 0, blk + 2, blk + 4, opsl);
    qs_to_opqs(blk + 16, opqs);
}

// Q5_K raw block (176B: d(2) dmin(2) sc(12) qh(32) qs(128)) -> opqs(128B) + opqh(32B) + opsl(16B).
inline void shuffle_q5k_block(const uint8_t* blk, uint8_t* opqs, uint8_t* opqh, uint8_t* opsl) {
    scale_min_to_psl(blk + 0, blk + 2, blk + 4, opsl);
    qs_to_opqs(blk + 48, opqs);

    const uint8_t* qh = blk + 16;
    std::memset(opqh, 0, 32);
    // opqh[j*4+i4] = OR_{s=0..7} ( ((qh[s*4+i4] >> j) & 1) << s ), j=0..7 (sub-block), i4=0..3
    for (int pos = 0; pos < 32; ++pos) {
        const int s = pos / 4, i4 = pos % 4;
        const uint8_t byte = qh[pos];
        for (int j = 0; j < 8; ++j) {
            const uint8_t bit = (byte >> j) & 1;
            opqh[j * 4 + i4] = static_cast<uint8_t>(opqh[j * 4 + i4] | (bit << s));
        }
    }
}

// Q6_K raw block (210B: ql(128) qh(64) sc(16,int8) d(2,f16)) -> opql(128B) + opqh(64B) + ps(16B) +
// pd(2B). ps/pd are a straight byte copy (int8 scales / f16 super-scale, unchanged).
inline void shuffle_q6k_block(const uint8_t* blk, uint8_t* opql, uint8_t* opqh, uint8_t* ps, uint8_t* pd) {
    const uint8_t* ql = blk;
    const uint8_t* qh = blk + 128;
    std::memcpy(ps, blk + 192, 16);
    std::memcpy(pd, blk + 208, 2);

    uint8_t rw_flat[256];
    for (int half = 0; half < 2; ++half) {
        const uint8_t* ql_h = ql + half * 64;
        const uint8_t* qh_h = qh + half * 32;
        uint8_t* rw = rw_flat + half * 128;
        for (int p = 0; p < 64; ++p) {
            rw[p] = ql_h[p] & 0x0F;
            rw[64 + p] = (ql_h[p] >> 4) & 0x0F;
        }
        for (int p = 0; p < 32; ++p) {
            const uint8_t qb = qh_h[p];
            rw[p] = static_cast<uint8_t>(rw[p] + ((qb & 0x03) << 4));
            rw[32 + p] = static_cast<uint8_t>(rw[32 + p] + ((qb & 0x0C) << 2));
            rw[64 + p] = static_cast<uint8_t>(rw[64 + p] + (qb & 0x30));
            rw[96 + p] = static_cast<uint8_t>(rw[96 + p] + ((qb & 0xC0) >> 2));
        }
    }

    // opql[j*16+i] = (rw_flat[j*32+i]&0xF) | ((rw_flat[j*32+16+i]&0xF)<<4), j=0..7, i=0..15
    for (int j = 0; j < 8; ++j) {
        for (int i = 0; i < 16; ++i) {
            const uint8_t lo = rw_flat[j * 32 + i] & 0x0F;
            const uint8_t hi = rw_flat[j * 32 + 16 + i] & 0x0F;
            opql[j * 16 + i] = static_cast<uint8_t>(lo | (hi << 4));
        }
    }
    // opqh[g*4+k] = ((rw_flat[g*16+k]&0x30)>>4) | ((rw_flat[g*16+4+k]&0x30)>>2)
    //             | (rw_flat[g*16+8+k]&0x30) | ((rw_flat[g*16+12+k]&0x30)<<2), g=0..15, k=0..3
    for (int g = 0; g < 16; ++g) {
        for (int k = 0; k < 4; ++k) {
            const uint8_t a = rw_flat[g * 16 + 0 + k] & 0x30;
            const uint8_t b = rw_flat[g * 16 + 4 + k] & 0x30;
            const uint8_t c = rw_flat[g * 16 + 8 + k] & 0x30;
            const uint8_t e = rw_flat[g * 16 + 12 + k] & 0x30;
            opqh[g * 4 + k] = static_cast<uint8_t>((a >> 4) | (b >> 2) | c | (e << 2));
        }
    }
}

// ---- Stage B: shuffled per-block sections -> transposed "SG" (sub-group column-major) layout.
// One work-group ("row-group") covers kSgOPG=16 consecutive output rows; within a row-group all
// lanes' bytes for a given (block, sub-block, chunk) are stored contiguously so a single
// intel_sub_group_block_read delivers one 4-byte word per lane. Identical to pack_qk_pqs_sg /
// pack_qk_psl_sg / pack_q5k_pqh_sg / pack_q6k_pqh_sg / pack_q6k_ps_sg / pack_q6k_pd_sg.

// 128-byte "qs"/"ql" section (Q4_K pqs, Q5_K pqs, Q6_K pql): dst[(((h*nbpr+bid)*8+j)*4+kc)*OPG*4
// + lid*4 + kb] = src[j*16 + kc*4 + kb], j=0..7 (sub-block), kc=0..3 (chunk), kb=0..3 (byte).
inline void pack_qs128_sg(const uint8_t* src, uint8_t* dst_group_base, int lid) {
    for (int j = 0; j < 8; ++j) {
        for (int kc = 0; kc < 4; ++kc) {
            uint8_t* dst = dst_group_base + (size_t)(j * 4 + kc) * kSgOPG * 4 + (size_t)lid * 4;
            const uint8_t* src4 = src + j * 16 + kc * 4;
            dst[0] = src4[0];
            dst[1] = src4[1];
            dst[2] = src4[2];
            dst[3] = src4[3];
        }
    }
}

// 16-byte psl "SoA seed" section (Q4_K / Q5_K): dst layout per (h,bid) group (256 bytes) =
// [sl_u32 x OPG=64B | ml_u32 x OPG=64B | sh_u16 x OPG=32B | mh_u16 x OPG=32B | d_u16 x OPG=32B |
//  dmin_u16 x OPG=32B].
inline void pack_psl16_sg(const uint8_t* src, uint8_t* dst_group_base, int lid) {
    std::memcpy(dst_group_base + lid * 4, src + 0, 4);
    std::memcpy(dst_group_base + 64 + lid * 4, src + 4, 4);
    std::memcpy(dst_group_base + 128 + lid * 2, src + 8, 2);
    std::memcpy(dst_group_base + 160 + lid * 2, src + 10, 2);
    std::memcpy(dst_group_base + 192 + lid * 2, src + 12, 2);
    std::memcpy(dst_group_base + 224 + lid * 2, src + 14, 2);
}

// 32-byte Q5_K qh section: dst[((h*nbpr+bid)*8+j)*OPG + lid)*4 + b] = src[j*4+b].
inline void pack_q5k_pqh32_sg(const uint8_t* src, uint8_t* dst_group_base, int lid) {
    for (int j = 0; j < 8; ++j) {
        uint8_t* dst = dst_group_base + (size_t)(j * kSgOPG + lid) * 4;
        const uint8_t* src4 = src + j * 4;
        dst[0] = src4[0];
        dst[1] = src4[1];
        dst[2] = src4[2];
        dst[3] = src4[3];
    }
}

// 64-byte Q6_K qh section: dst[(((h*nbpr+bid)*8+j)*2+c)*OPG + lid)*4 + b] = src[j*8+c*4+b].
inline void pack_q6k_pqh64_sg(const uint8_t* src, uint8_t* dst_group_base, int lid) {
    for (int j = 0; j < 8; ++j) {
        for (int c = 0; c < 2; ++c) {
            uint8_t* dst = dst_group_base + (size_t)((j * 2 + c) * kSgOPG + lid) * 4;
            const uint8_t* src4 = src + j * 8 + c * 4;
            dst[0] = src4[0];
            dst[1] = src4[1];
            dst[2] = src4[2];
            dst[3] = src4[3];
        }
    }
}

// 16-byte Q6_K int8-scale section: dst[si*OPG + lid] = src[si], si=0..15.
inline void pack_q6k_ps16_sg(const uint8_t* src, uint8_t* dst_group_base, int lid) {
    for (int si = 0; si < 16; ++si) {
        dst_group_base[si * kSgOPG + lid] = src[si];
    }
}

// 2-byte Q6_K f16 super-scale: dst[lid*2 + b] = src[b], b=0..1.
inline void pack_q6k_pd2_sg(const uint8_t* src, uint8_t* dst_group_base, int lid) {
    dst_group_base[lid * 2 + 0] = src[0];
    dst_group_base[lid * 2 + 1] = src[1];
}

// Repack ONE expert's raw GGUF weight bytes [N, K] (block-quantized, block_elem=256) into the
// transposed "SG" layout consumed by moe_gguf_sg_gemv.cl. `code` selects Q4_K(1)/Q5_K(2)/Q6_K(3).
// expert_size (== output buffer size for this expert) is unchanged from the raw layout.
void pack_expert_sg(int code, const uint8_t* raw_expert, uint8_t* dst_expert, size_t N, size_t nprow) {
    const size_t OPG = kSgOPG;
    const size_t nrg = N / OPG;
    const size_t num_blocks = N * nprow;

    size_t sec_qs = 0, sec_qh = 0, sec_psl_ps = 0;
    size_t blk_bytes = 0;
    switch (code) {
    case 1:  // Q4_K
        blk_bytes = 144;
        sec_qs = num_blocks * 128;
        sec_psl_ps = num_blocks * 16;
        break;
    case 2:  // Q5_K
        blk_bytes = 176;
        sec_qs = num_blocks * 128;
        sec_qh = num_blocks * 32;
        sec_psl_ps = num_blocks * 16;
        break;
    case 3:  // Q6_K
        blk_bytes = 210;
        sec_qs = num_blocks * 128;
        sec_qh = num_blocks * 64;
        sec_psl_ps = num_blocks * 16;
        break;
    default:
        return;
    }

    uint8_t* dst_qs = dst_expert;
    uint8_t* dst_qh = dst_qs + sec_qs;
    uint8_t* dst_psl_ps = dst_qh + sec_qh;
    uint8_t* dst_pd = dst_psl_ps + sec_psl_ps;

    ov::parallel_for(nrg * nprow, [&](size_t hb) {
        const size_t h = hb / nprow;
        const size_t bid = hb % nprow;
        const size_t group_idx = h * nprow + bid;  // (h*nbpr+bid), used by every section's stride

        uint8_t* qs_group = dst_qs + group_idx * OPG * 128;
        uint8_t* qh_group = dst_qh + group_idx * OPG * (code == 2 ? 32 : (code == 3 ? 64 : 0));
        uint8_t* psl_ps_group = dst_psl_ps + group_idx * OPG * 16;
        uint8_t* pd_group = dst_pd + group_idx * OPG * 2;

        for (size_t lid = 0; lid < OPG; ++lid) {
            const size_t n = h * OPG + lid;
            const size_t blk = n * nprow + bid;
            const uint8_t* raw_blk = raw_expert + blk * blk_bytes;

            if (code == 1) {
                uint8_t opqs[128], opsl[16];
                shuffle_q4k_block(raw_blk, opqs, opsl);
                pack_qs128_sg(opqs, qs_group, static_cast<int>(lid));
                pack_psl16_sg(opsl, psl_ps_group, static_cast<int>(lid));
            } else if (code == 2) {
                uint8_t opqs[128], opqh[32], opsl[16];
                shuffle_q5k_block(raw_blk, opqs, opqh, opsl);
                pack_qs128_sg(opqs, qs_group, static_cast<int>(lid));
                pack_q5k_pqh32_sg(opqh, qh_group, static_cast<int>(lid));
                pack_psl16_sg(opsl, psl_ps_group, static_cast<int>(lid));
            } else {  // code == 3
                uint8_t opql[128], opqh[64], ps[16], pd[2];
                shuffle_q6k_block(raw_blk, opql, opqh, ps, pd);
                pack_qs128_sg(opql, qs_group, static_cast<int>(lid));
                pack_q6k_pqh64_sg(opqh, qh_group, static_cast<int>(lid));
                pack_q6k_ps16_sg(ps, psl_ps_group, static_cast<int>(lid));
                pack_q6k_pd2_sg(pd, pd_group, static_cast<int>(lid));
            }
        }
    });
}

// Returns the Q4_K/Q5_K/Q6_K decode code (1/2/3) for the SG repack, or 0 if `t` is not one of
// these three types (e.g. Q4_0/Q8_0, handled by the existing raw-GGUF-block decode kernel /
// moe_gguf_sg_gemv.cl's dedicated Q8_0 shared-expert kernels -- see moe_3gemm_swiglu_opt.cpp).
int moe_sg_decode_code(const ov::element::Type& t) {
    if (t == ov::element::gguf_q4_k)
        return 1;
    if (t == ov::element::gguf_q5_k)
        return 2;
    if (t == ov::element::gguf_q6_k)
        return 3;
    return 0;
}

// N-grouped byte-transpose of one Q4_K/Q5_K/Q6_K GGUF weight Constant (routed [E, N, K] or shared
// [N, K]) into the SG layout. Returns nullptr if the Constant cannot be packed (N not a multiple of
// kSgOPG, or K not a whole number of 256-element blocks).
std::shared_ptr<ov::op::v0::Constant> pack_moe_weight_sg(const std::shared_ptr<ov::op::v0::Constant>& w_const) {
    const ov::element::Type wt = w_const->get_element_type();
    const int code = moe_sg_decode_code(wt);
    if (code == 0) {
        return nullptr;
    }
    const auto& shape = w_const->get_shape();
    size_t E, N, K;
    if (shape.size() == 3) {
        E = shape[0];
        N = shape[1];
        K = shape[2];
    } else if (shape.size() == 2) {
        E = 1;
        N = shape[0];
        K = shape[1];
    } else {
        return nullptr;
    }

    constexpr size_t kBlockElem = 256;  // Q4_K/Q5_K/Q6_K all use a 256-element super-block.
    if (K % kBlockElem != 0 || N % kSgOPG != 0) {
        return nullptr;
    }
    const size_t nprow = K / kBlockElem;
    const size_t expert_bytes = w_const->get_byte_size() / std::max<size_t>(E, 1);
    const size_t total_bytes = w_const->get_byte_size();
    if (total_bytes != E * expert_bytes) {
        return nullptr;
    }

    const auto* src = static_cast<const uint8_t*>(w_const->get_data_ptr());
    auto dst_buf = std::make_shared<ov::AlignedBuffer>(total_bytes, 64);
    auto* dst = static_cast<uint8_t*>(dst_buf->get_ptr());

    ov::parallel_for(E, [&](size_t e) {
        pack_expert_sg(code, src + e * expert_bytes, dst + e * expert_bytes, N, nprow);
    });

    auto packed = std::make_shared<ov::op::v0::Constant>(wt, shape, dst_buf);
    packed->set_friendly_name(w_const->get_friendly_name());
    ov::copy_runtime_info(w_const, packed);
    return packed;
}

// Full GGUF MoE decode-type code (mirrors moe_3gemm_swiglu_opt.cpp's moe_gguf_decode_code):
// 1=Q4_K, 2=Q5_K, 3=Q6_K, 4=Q8_0. 0 = not a supported GGUF MoE block. Unlike
// moe_sg_decode_code() above (which only covers the three SG-repackable K-types), this also
// recognizes Q8_0 so the shared-expert Q8_0 packing gate below can mirror
// moe_3gemm_swiglu_opt.cpp's shared_expert_sg_ok / _use_gguf_moe_sg checks exactly.
int moe_gguf_full_decode_code(const ov::element::Type& t) {
    const int code = moe_sg_decode_code(t);
    if (code != 0) {
        return code;
    }
    if (t == ov::element::gguf_q8_0) {
        return 4;
    }
    return 0;
}

// ============================================================================================
// Shared-expert Q8_0 SG repack: identical byte transform to RepackGGUFWeightsShuffle's Q8_0
// branch (repack_gguf_weights.cpp, is_small && is_q8_0) and to the layout fc_gguf_q8_0_sg.cl
// decodes -- see moe_gguf_sg_gemv.cl's "Shared-expert Q8_0 kernels" header comment for the
// full byte-layout description. `src` is one native raw-GGUF Q8_0 weight matrix [N, K]
// (34 bytes/block = f16 d + int8 qs[32]); `dst` (same total byte size) receives the SG-packed
// [pqs | pd] layout consumed by shared_gate_up_q8_0 / shared_down_merge_q8_0.
// ============================================================================================
void pack_shared_q8_0_sg(const uint8_t* src, uint8_t* dst, size_t N, size_t K) {
    constexpr size_t kBlkBytes = 34;      // native q8_0 block: half d + int8 qs[32]
    const size_t nbpr_native = K / 32;    // native 32-wide blocks per row
    const size_t nbpr = K / 256;          // 256-elem super-blocks per row (== nbpr_native / 8)
    const size_t nrg = N / kSgOPG;
    const size_t off_pd = N * nbpr * 256;

    ov::parallel_for(nrg * nbpr, [&](size_t hb) {
        const size_t h = hb / nbpr;
        const size_t bid = hb % nbpr;
        uint8_t* pqs_entry = dst + (h * nbpr + bid) * kSgOPG * 256;
        uint8_t* pd_entry = dst + off_pd + (h * nbpr + bid) * kSgOPG * 16;

        for (size_t lid = 0; lid < kSgOPG; ++lid) {
            const size_t row = h * kSgOPG + lid;
            uint8_t pqs[256];
            uint8_t pd[16];
            for (size_t j = 0; j < 8; ++j) {
                const size_t nb = bid * 8 + j;
                const uint8_t* blk = src + (row * nbpr_native + nb) * kBlkBytes;
                pd[j * 2 + 0] = blk[0];
                pd[j * 2 + 1] = blk[1];
                std::memcpy(pqs + j * 32, blk + 2, 32);
            }
            // chunk-interleave pqs (256 bytes = 64 x 4-byte chunks), matching pack_qs128_sg's
            // scatter pattern (same as RepackGGUFWeightsShuffle's sg_scatter_chunks for Q8_0).
            for (size_t c = 0; c < 64; ++c) {
                std::memcpy(pqs_entry + c * kSgOPG * 4 + lid * 4, pqs + c * 4, 4);
            }
            // pd SoA: field j (0..7) at j*kSgOPG*2 + lid*2.
            for (size_t j = 0; j < 8; ++j) {
                std::memcpy(pd_entry + j * kSgOPG * 2 + lid * 2, pd + j * 2, 2);
            }
        }
    });
}

}  // namespace

RepackGGUFMoEWeights::RepackGGUFMoEWeights() {
    using namespace ov::pass::pattern;

    auto moe_m = wrap_type<ov::op::internal::MOECompressed>();

    ov::matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](ov::pass::pattern::Matcher& m) {
        if (!moe_sg_enabled()) {
            return false;
        }
        auto moe = ov::as_type_ptr<ov::op::internal::MOECompressed>(m.get_match_root());
        if (!moe) {
            return false;
        }

        // Collect the GGUF-block weight Constants among the op inputs (routed gate/up/down 3D +
        // shared gate/up/down 2D). Scale/zp inputs are empty non-GGUF placeholders — skipped. Only
        // Q4_K/Q5_K/Q6_K (moe_sg_decode_code() != 0) are repacked here into the SG layout via
        // pack_moe_weight_sg(); routed Q4_0/Q8_0 are left untouched in their raw per-row GGUF-block
        // layout (decoded by the raw-GGUF-block batched-GEMV kernels, moe_3gemm_swiglu_mlp.cl).
        // The shared expert's Q8_0 gate/up/down weights (if present) get their OWN SG repack below
        // (pack_shared_q8_0_sg()) into the layout moe_gguf_sg_gemv.cl's shared_gate_up_q8_0 /
        // shared_down_merge_q8_0 kernels decode with intel_sub_group_block_read8 -- see those
        // kernels' header comment and moe_3gemm_swiglu_opt.cpp's shared_expert_sg_ok check.
        std::vector<size_t> gguf_inputs;
        for (size_t i = 0; i < moe->get_input_size(); ++i) {
            const auto in = moe->input_value(i);
            if (!in.get_element_type().is_gguf_block()) {
                continue;
            }
            auto c = ov::as_type_ptr<ov::op::v0::Constant>(in.get_node_shared_ptr());
            if (!c) {
                continue;
            }
            if (moe_sg_decode_code(c->get_element_type()) == 0) {
                continue;  // not SG-packable (e.g. Q4_0/Q8_0) -- leave in raw GGUF-block layout
            }
            const auto& s = c->get_shape();
            const size_t N = (s.size() == 3) ? s[1] : (s.size() == 2 ? s[0] : 0);
            if (N == 0 || N % kSgOPG != 0) {
                return false;  // inconsistent packability -> pack nothing for this op
            }
            gguf_inputs.push_back(i);
        }
        // Shared-expert Q8_0 gate/up/down: mirror moe_3gemm_swiglu_opt.cpp's shared_expert_sg_ok
        // gate exactly -- only pack when ALL THREE shared weights are GGUF Q8_0 (that impl only
        // enables the SG decode path for a shared expert under that same condition; any other
        // shared-expert weight type must be left in its native layout since the impl will fall
        // back to the raw-GGUF-block batched-GEMV kernel, which expects the raw layout for every
        // GGUF weight, shared or routed).
        constexpr size_t kSharedGateWeightIdx = 12, kSharedUpWeightIdx = 15, kSharedDownWeightIdx = 18;
        constexpr size_t kW0Idx = 3, kW1Idx = 6, kW2Idx = 9;
        std::vector<size_t> gguf_q8_0_shared_inputs;
        if (moe->get_input_size() > kSharedDownWeightIdx) {
            const auto sg_dt = moe->input_value(kSharedGateWeightIdx).get_element_type();
            const auto su_dt = moe->input_value(kSharedUpWeightIdx).get_element_type();
            const auto sd_dt = moe->input_value(kSharedDownWeightIdx).get_element_type();
            const bool shared_all_q8_0 = moe_gguf_full_decode_code(sg_dt) == 4 && moe_gguf_full_decode_code(su_dt) == 4 &&
                                         moe_gguf_full_decode_code(sd_dt) == 4;
            // Also require the routed weights to be SG-decodable. This MUST match _use_gguf_moe_sg
            // in moe_3gemm_swiglu_opt.cpp exactly, which uses moe_gguf_sg_routable() (codes 1..3 =
            // Q4_K/Q5_K/Q6_K), NOT a plain "!= 0" (a plain "!= 0" also matches routed Q8_0/Q4_0, for
            // which the impl actually falls back to the raw-GGUF-block kernel -- SG-packing the
            // shared Q8_0 weights then would make BOTH the decode kernel AND the prefill transcode
            // read them from the wrong layout).
            auto routed_sg = [](const ov::element::Type& t) {
                const int c = moe_gguf_full_decode_code(t);
                return c >= 1 && c <= 3;
            };
            const bool routed_ok = routed_sg(moe->input_value(kW0Idx).get_element_type()) &&
                                   routed_sg(moe->input_value(kW1Idx).get_element_type()) &&
                                   routed_sg(moe->input_value(kW2Idx).get_element_type());
            if (shared_all_q8_0 && routed_ok) {
                for (size_t idx : {kSharedGateWeightIdx, kSharedUpWeightIdx, kSharedDownWeightIdx}) {
                    auto c = ov::as_type_ptr<ov::op::v0::Constant>(moe->input_value(idx).get_node_shared_ptr());
                    if (!c) {
                        continue;
                    }
                    const auto& s = c->get_shape();
                    if (s.size() != 2 || s[0] % kSgOPG != 0 || s[1] % 256 != 0) {
                        continue;  // inconsistent shape -> leave this weight raw
                    }
                    gguf_q8_0_shared_inputs.push_back(idx);
                }
            }
        }

        if (gguf_inputs.empty() && gguf_q8_0_shared_inputs.empty()) {
            return false;
        }

        bool changed = false;
        for (size_t idx : gguf_inputs) {
            auto c = ov::as_type_ptr<ov::op::v0::Constant>(moe->input_value(idx).get_node_shared_ptr());
            auto packed = pack_moe_weight_sg(c);
            if (!packed) {
                continue;
            }
            moe->input(idx).replace_source_output(packed->output(0));
            changed = true;
        }
        for (size_t idx : gguf_q8_0_shared_inputs) {
            auto c = ov::as_type_ptr<ov::op::v0::Constant>(moe->input_value(idx).get_node_shared_ptr());
            const auto& s = c->get_shape();  // [N, K]
            const size_t N = s[0], K = s[1];
            const size_t total_bytes = c->get_byte_size();
            const auto* src = static_cast<const uint8_t*>(c->get_data_ptr());
            auto dst_buf = std::make_shared<ov::AlignedBuffer>(total_bytes, 64);
            auto* dst = static_cast<uint8_t*>(dst_buf->get_ptr());
            pack_shared_q8_0_sg(src, dst, N, K);

            auto packed = std::make_shared<ov::op::v0::Constant>(c->get_element_type(), s, dst_buf);
            packed->set_friendly_name(c->get_friendly_name());
            ov::copy_runtime_info(c, packed);
            moe->input(idx).replace_source_output(packed->output(0));
            changed = true;
        }
        return changed;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(moe_m, "RepackGGUFMoEWeights");
    this->register_matcher(m, callback);
}

}  // namespace ov::intel_gpu
