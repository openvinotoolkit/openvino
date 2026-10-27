// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "repack_gguf_weights.hpp"

#include <cstdlib>
#include <cstring>
#include <memory>

#include "intel_gpu/op/fully_connected_compressed.hpp"
#include "openvino/core/parallel.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "openvino/runtime/aligned_buffer.hpp"
#include "openvino/util/pp.hpp"

namespace ov::intel_gpu {

namespace {

// SG (sub-group) width == output-row group size. Must match OPG / SG_SIZE in fc_gguf_q4k_sg.cl /
// fc_gguf_q5k_sg.cl / fc_gguf_q6k_sg.cl and the dispatch in FCGGUFOptImpl.
constexpr size_t kSG = 16;

// Weight-shuffle gate: opt-out via OV_GPU_GGUF_SHUFFLE=0. Default ON. Mirrored by FCGGUFOptImpl so the
// transform and the impl agree which Q4_K/Q5_K/Q6_K nodes are shuffled.
bool shuffle_enabled() {
    if (const char* env = std::getenv("OV_GPU_GGUF_SHUFFLE")) {
        return std::atol(env) != 0;
    }
    return true;
}

// Q4_K / Q5_K ggml get_scale_min_k4: decode the 12-byte packed scales into 8 (scale, min) 6-bit values.
inline void q4k_scale_min(int j, const uint8_t* q, uint8_t& d, uint8_t& m) {
    if (j < 4) {
        d = q[j] & 63;
        m = q[j + 4] & 63;
    } else {
        d = static_cast<uint8_t>((q[j + 4] & 0x0F) | ((q[j - 4] >> 6) << 4));
        m = static_cast<uint8_t>((q[j + 4] >> 4) | ((q[j] >> 6) << 4));
    }
}

// ---- Step 1: shuffle one native Q4_K block (144 B) -> pqs[128] + psl[16] (row-major). ----
// Mirrors shuffle_q4k() in q4k_gemv/test_gemv_sg_kernels.py.
void shuffle_q4k_block(const uint8_t* blk, uint8_t* pqs, uint8_t* psl) {
    const uint8_t* sc = blk + 4;   // scales[12]
    const uint8_t* qs = blk + 16;  // qs[128]

    // Un-nibble the 256 quants into rw[256]: within each 32-elem sub-block low nibbles first (0..31)
    // then high (32..63) — matching the reference dequant order.
    uint8_t rw[256];
    for (int j = 0; j < 4; ++j) {
        for (int k = 0; k < 32; ++k) {
            rw[j * 64 + k] = qs[j * 32 + k] & 0x0F;
            rw[j * 64 + 32 + k] = (qs[j * 32 + k] >> 4) & 0x0F;
        }
    }

    // Decode the 8 (scale, min) 6-bit values.
    uint8_t rs[8], rm[8];
    for (int j = 0; j < 8; ++j) {
        q4k_scale_min(j, sc, rs[j], rm[j]);
    }

    // pqs: 8 sub-blocks x 16 bytes; byte = low nibble(pos k) | (high nibble(pos 16+k) << 4).
    for (int j = 0; j < 8; ++j) {
        for (int k = 0; k < 16; ++k) {
            pqs[j * 16 + k] = static_cast<uint8_t>((rw[j * 32 + k] & 0xF) | ((rw[j * 32 + 16 + k] & 0xF) << 4));
        }
    }

    // psl: sl(4) ml(4) sh(2) mh(2) d(2) dmin(2). sl/ml hold the 8 low-4-bit; sh/mh the 8 high-2-bit.
    psl[0] = static_cast<uint8_t>((rs[0] & 0xF) | ((rs[1] & 0xF) << 4));
    psl[1] = static_cast<uint8_t>((rs[2] & 0xF) | ((rs[3] & 0xF) << 4));
    psl[2] = static_cast<uint8_t>((rs[4] & 0xF) | ((rs[5] & 0xF) << 4));
    psl[3] = static_cast<uint8_t>((rs[6] & 0xF) | ((rs[7] & 0xF) << 4));
    psl[4] = static_cast<uint8_t>((rm[0] & 0xF) | ((rm[1] & 0xF) << 4));
    psl[5] = static_cast<uint8_t>((rm[2] & 0xF) | ((rm[3] & 0xF) << 4));
    psl[6] = static_cast<uint8_t>((rm[4] & 0xF) | ((rm[5] & 0xF) << 4));
    psl[7] = static_cast<uint8_t>((rm[6] & 0xF) | ((rm[7] & 0xF) << 4));
    psl[8]  = static_cast<uint8_t>(((rs[0] & 0x30) >> 4) | ((rs[1] & 0x30) >> 2) | (rs[2] & 0x30) | ((rs[3] & 0x30) << 2));
    psl[9]  = static_cast<uint8_t>(((rs[4] & 0x30) >> 4) | ((rs[5] & 0x30) >> 2) | (rs[6] & 0x30) | ((rs[7] & 0x30) << 2));
    psl[10] = static_cast<uint8_t>(((rm[0] & 0x30) >> 4) | ((rm[1] & 0x30) >> 2) | (rm[2] & 0x30) | ((rm[3] & 0x30) << 2));
    psl[11] = static_cast<uint8_t>(((rm[4] & 0x30) >> 4) | ((rm[5] & 0x30) >> 2) | (rm[6] & 0x30) | ((rm[7] & 0x30) << 2));
    // d, dmin (fp16, little-endian) copied verbatim.
    psl[12] = blk[0];
    psl[13] = blk[1];
    psl[14] = blk[2];
    psl[15] = blk[3];
}

// ---- Step 1: shuffle one native Q5_K block (176 B) -> pqs[128] + pqh[32] + psl[16] (row-major). ----
// Q5_K == Q4_K plus one extra high bit per weight (weight range 0..31). pqs / psl match Q4_K exactly;
// pqh packs the high bit of weight wi (0..31) within sub-block j into byte (wi%4), bit (wi/4) of the
// sub-block's 4-byte word (matches Q5K_ACC_CHUNK / tq_decode_shuffle_q5k). Native block layout:
//   d(2) dmin(2) scales[12] qh[32] qs[128].
void shuffle_q5k_block(const uint8_t* blk, uint8_t* pqs, uint8_t* pqh, uint8_t* psl) {
    const uint8_t* sc = blk + 4;   // scales[12]
    const uint8_t* qh = blk + 16;  // qh[32]  (high bits)
    const uint8_t* qs = blk + 48;  // qs[128] (low 4-bit quants)

    // Reconstruct the 5-bit quants rw[256] (low 4-bit + high bit merged, range 0..31), in the
    // reference dequant order (sub-block s of 32 weights, s=0..7).
    uint8_t rw[256];
    for (int jj = 0; jj < 4; ++jj) {
        const uint8_t* ql = qs + jj * 32;
        const uint8_t u1 = static_cast<uint8_t>(1u << (2 * jj));
        const uint8_t u2 = static_cast<uint8_t>(2u << (2 * jj));
        for (int l = 0; l < 32; ++l) {
            rw[(2 * jj) * 32 + l]     = static_cast<uint8_t>((ql[l] & 0x0F) + ((qh[l] & u1) ? 16 : 0));
            rw[(2 * jj + 1) * 32 + l] = static_cast<uint8_t>((ql[l] >> 4)   + ((qh[l] & u2) ? 16 : 0));
        }
    }

    // Decode the 8 (scale, min) 6-bit values (identical 12-byte packing to Q4_K).
    uint8_t rs[8], rm[8];
    for (int j = 0; j < 8; ++j) {
        q4k_scale_min(j, sc, rs[j], rm[j]);
    }

    // pqs: 8 sub-blocks x 16 bytes; byte = low nibble(pos k) | (low nibble(pos 16+k) << 4).
    for (int j = 0; j < 8; ++j) {
        for (int k = 0; k < 16; ++k) {
            pqs[j * 16 + k] = static_cast<uint8_t>((rw[j * 32 + k] & 0xF) | ((rw[j * 32 + 16 + k] & 0xF) << 4));
        }
    }

    // pqh: 8 sub-blocks x 4 bytes; high bit of weight wi (0..31) at byte (wi%4), bit (wi/4).
    for (int j = 0; j < 32; ++j) {
        pqh[j] = 0;
    }
    for (int j = 0; j < 8; ++j) {
        for (int wi = 0; wi < 32; ++wi) {
            const uint8_t hb = static_cast<uint8_t>((rw[j * 32 + wi] >> 4) & 1);
            pqh[j * 4 + (wi & 3)] |= static_cast<uint8_t>(hb << (wi >> 2));
        }
    }

    // psl: sl(4) ml(4) sh(2) mh(2) d(2) dmin(2) — identical to Q4_K.
    psl[0] = static_cast<uint8_t>((rs[0] & 0xF) | ((rs[1] & 0xF) << 4));
    psl[1] = static_cast<uint8_t>((rs[2] & 0xF) | ((rs[3] & 0xF) << 4));
    psl[2] = static_cast<uint8_t>((rs[4] & 0xF) | ((rs[5] & 0xF) << 4));
    psl[3] = static_cast<uint8_t>((rs[6] & 0xF) | ((rs[7] & 0xF) << 4));
    psl[4] = static_cast<uint8_t>((rm[0] & 0xF) | ((rm[1] & 0xF) << 4));
    psl[5] = static_cast<uint8_t>((rm[2] & 0xF) | ((rm[3] & 0xF) << 4));
    psl[6] = static_cast<uint8_t>((rm[4] & 0xF) | ((rm[5] & 0xF) << 4));
    psl[7] = static_cast<uint8_t>((rm[6] & 0xF) | ((rm[7] & 0xF) << 4));
    psl[8]  = static_cast<uint8_t>(((rs[0] & 0x30) >> 4) | ((rs[1] & 0x30) >> 2) | (rs[2] & 0x30) | ((rs[3] & 0x30) << 2));
    psl[9]  = static_cast<uint8_t>(((rs[4] & 0x30) >> 4) | ((rs[5] & 0x30) >> 2) | (rs[6] & 0x30) | ((rs[7] & 0x30) << 2));
    psl[10] = static_cast<uint8_t>(((rm[0] & 0x30) >> 4) | ((rm[1] & 0x30) >> 2) | (rm[2] & 0x30) | ((rm[3] & 0x30) << 2));
    psl[11] = static_cast<uint8_t>(((rm[4] & 0x30) >> 4) | ((rm[5] & 0x30) >> 2) | (rm[6] & 0x30) | ((rm[7] & 0x30) << 2));
    // d, dmin (fp16, little-endian) copied verbatim.
    psl[12] = blk[0];
    psl[13] = blk[1];
    psl[14] = blk[2];
    psl[15] = blk[3];
}

// ---- Step 1: shuffle one native Q6_K block (210 B) -> pql[128] + pqh[64] + ps[16] + pd[2]. ----
// Mirrors shuffle_q6k() in q4k_gemv/test_gemv_sg_kernels.py.
void shuffle_q6k_block(const uint8_t* blk, uint8_t* pql, uint8_t* pqh, uint8_t* ps, uint8_t* pd) {
    const uint8_t* ql_r = blk;        // ql[128]
    const uint8_t* qh_r = blk + 128;  // qh[64]
    const uint8_t* sc_r = blk + 192;  // scales[16] (int8)

    // Reconstruct the 6-bit quants rw[256] (low 4-bit + high 2-bit merged), in the reference order.
    uint8_t rw[256];
    for (int j = 0; j < 2; ++j) {
        for (int k = 0; k < 64; ++k) {
            rw[j * 128 + k] = ql_r[j * 64 + k] & 0x0F;
            rw[j * 128 + 64 + k] = (ql_r[j * 64 + k] >> 4) & 0x0F;
        }
        for (int k = 0; k < 32; ++k) {
            rw[j * 128 + k] = static_cast<uint8_t>(rw[j * 128 + k] + ((qh_r[j * 32 + k] & 0x03) << 4));
            rw[j * 128 + 32 + k] = static_cast<uint8_t>(rw[j * 128 + 32 + k] + ((qh_r[j * 32 + k] & 0x0C) << 2));
            rw[j * 128 + 64 + k] = static_cast<uint8_t>(rw[j * 128 + 64 + k] + (qh_r[j * 32 + k] & 0x30));
            rw[j * 128 + 96 + k] = static_cast<uint8_t>(rw[j * 128 + 96 + k] + ((qh_r[j * 32 + k] & 0xC0) >> 2));
        }
    }

    // pql: 8 sub-blocks x 16 bytes; byte = low nibble(pos k) | (high nibble(pos 16+k) << 4).
    for (int j = 0; j < 8; ++j) {
        for (int k = 0; k < 16; ++k) {
            pql[j * 16 + k] = static_cast<uint8_t>((rw[j * 32 + k] & 0xF) | ((rw[j * 32 + 16 + k] & 0xF) << 4));
        }
    }

    // pqh: 16 groups x 4 bytes; each byte packs the 2 high bits of 4 positions (k, k+4, k+8, k+12).
    for (int j = 0; j < 16; ++j) {
        for (int k = 0; k < 4; ++k) {
            pqh[j * 4 + k] = static_cast<uint8_t>(((rw[j * 16 + k] & 0x30) >> 4) | ((rw[j * 16 + k + 4] & 0x30) >> 2) |
                                                  (rw[j * 16 + k + 8] & 0x30) | ((rw[j * 16 + k + 12] & 0x30) << 2));
        }
    }

    // ps: 16 int8 scales copied verbatim; pd: fp16 super-scale copied verbatim.
    std::memcpy(ps, sc_r, 16);
    pd[0] = blk[208];
    pd[1] = blk[209];
}

// ---- Step 2: SG-transpose a per-plane row-major buffer. ----
// For a plane whose per-block payload is `pbytes` bytes, lane `lid` (= row within group) holds the
// block bytes interleaved in 4-byte chunks: for byte offset `o`, chunk c = o/4 -> dst = c*kSG*4 +
// lid*4 (+ o%4). Matches pack_q4k_sg/pack_q5k_sg/pack_q6k_sg. `pbytes` must be a multiple of 4
// (pqs/pql=128, pqh(Q5_K)=32, pqh(Q6_K)=64).
inline void sg_scatter_chunks(uint8_t* entry, const uint8_t* src, size_t pbytes, size_t lid) {
    const size_t chunks = pbytes / 4;
    for (size_t c = 0; c < chunks; ++c) {
        std::memcpy(entry + c * kSG * 4 + lid * 4, src + c * 4, 4);
    }
}

}  // namespace

RepackGGUFWeightsShuffle::RepackGGUFWeightsShuffle() {
    using namespace ov::pass::pattern;

    auto fc_m = wrap_type<ov::intel_gpu::op::FullyConnectedCompressed>();

    ov::matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](ov::pass::pattern::Matcher& m) {
        auto fc = ov::as_type_ptr<ov::intel_gpu::op::FullyConnectedCompressed>(m.get_match_root());
        if (!fc) {
            return false;
        }

        if (!shuffle_enabled()) {
            return false;
        }
        // Weight is input(1). Q4_K, Q5_K, Q6_K (native 256-elem block) and the small-block formats
        // Q4_0, Q4_1, Q8_0 (native 32-elem block, grouped 8-to-a 256-elem super-block) get the shuffle
        // layout; every other format keeps its native block layout (read by the native float-decode GEMV).
        const auto w_out = fc->input_value(1);
        const auto et = w_out.get_element_type();
        const bool is_q4k = (et == ov::element::gguf_q4_k);
        const bool is_q5k = (et == ov::element::gguf_q5_k);
        const bool is_q6k = (et == ov::element::gguf_q6_k);
        const bool is_q4_0 = (et == ov::element::gguf_q4_0);
        const bool is_q4_1 = (et == ov::element::gguf_q4_1);
        const bool is_q8_0 = (et == ov::element::gguf_q8_0);
        const bool is_small = is_q4_0 || is_q4_1 || is_q8_0;
        if (!is_q4k && !is_q5k && !is_q6k && !is_small) {
            return false;
        }
        auto w_const = ov::as_type_ptr<ov::op::v0::Constant>(w_out.get_node_shared_ptr());
        if (!w_const) {
            return false;
        }
        const auto& shape = w_const->get_shape();  // [N, K]
        if (shape.size() != 2) {
            return false;
        }
        const size_t N = shape[0];
        const size_t K = shape[1];

        const ov::element::Type wt = et;
        const size_t be = wt.block_elem_count();  // 256 (K-formats) or 32 (small-block formats)
        const size_t bb = wt.block_byte_size();   // Q4_K 144 / Q5_K 176 / Q6_K 210 / Q4_0 18 / Q4_1 20 / Q8_0 34
        // SG grouping needs N % 16 == 0; K must be a whole number of 256-elem super-blocks (K-formats
        // use one native block per super-block; small-block formats group 8 native 32-elem blocks).
        if (N % kSG != 0) {
            return false;
        }
        if (is_small) {
            if (be != 32 || K % 256 != 0) {
                return false;
            }
        } else if (be != 256 || K % be != 0) {
            return false;
        }
        const size_t nbpr = K / 256;         // 256-elem super-blocks per row (== K-blocks for K-formats)
        const size_t nrg = N / kSG;          // row groups
        const size_t nbpr_native = K / be;   // native blocks per row

        const size_t total_bytes = w_const->get_byte_size();  // == N*nbpr_native*bb
        if (total_bytes != N * nbpr_native * bb) {
            return false;
        }

        const auto* src = static_cast<const uint8_t*>(w_const->get_data_ptr());
        auto dst_buf = std::make_shared<ov::AlignedBuffer>(total_bytes, 64);
        auto* dst = static_cast<uint8_t*>(dst_buf->get_ptr());

        // Per-plane SG-packed entry bases. Same total size as the native block layout.
        // Q4_K: pqs[128] + psl[16].  Q5_K: pqs[128] + pqh[32] + psl[16].  Q6_K: pql[128] + pqh[64] +
        // ps[16] + pd[2].
        if (is_q4k) {
            const size_t off_pqs = 0;
            const size_t off_psl = N * nbpr * 128;
            ov::parallel_for(nrg, [&](size_t h) {
                for (size_t bid = 0; bid < nbpr; ++bid) {
                    const size_t pqs_entry = off_pqs + (h * nbpr + bid) * kSG * 128;
                    const size_t psl_entry = off_psl + (h * nbpr + bid) * kSG * 16;
                    for (size_t lid = 0; lid < kSG; ++lid) {
                        const size_t row = h * kSG + lid;
                        const size_t blk = row * nbpr + bid;
                        const uint8_t* blk_src = src + blk * bb;

                        uint8_t pqs[128];
                        uint8_t psl[16];
                        shuffle_q4k_block(blk_src, pqs, psl);

                        sg_scatter_chunks(dst + pqs_entry, pqs, 128, lid);
                        // psl SoA fields: sl(4) ml(4) sh(2) mh(2) d(2) dmin(2).
                        uint8_t* pe = dst + psl_entry;
                        std::memcpy(pe + 0 + lid * 4, psl + 0, 4);    // sl
                        std::memcpy(pe + 64 + lid * 4, psl + 4, 4);   // ml
                        std::memcpy(pe + 128 + lid * 2, psl + 8, 2);  // sh
                        std::memcpy(pe + 160 + lid * 2, psl + 10, 2); // mh
                        std::memcpy(pe + 192 + lid * 2, psl + 12, 2); // d
                        std::memcpy(pe + 224 + lid * 2, psl + 14, 2); // dmin
                    }
                }
            });
        } else if (is_q5k) {
            const size_t off_pqs = 0;
            const size_t off_pqh = N * nbpr * 128;
            const size_t off_psl = off_pqh + N * nbpr * 32;
            ov::parallel_for(nrg, [&](size_t h) {
                for (size_t bid = 0; bid < nbpr; ++bid) {
                    const size_t pqs_entry = off_pqs + (h * nbpr + bid) * kSG * 128;
                    const size_t pqh_entry = off_pqh + (h * nbpr + bid) * kSG * 32;
                    const size_t psl_entry = off_psl + (h * nbpr + bid) * kSG * 16;
                    for (size_t lid = 0; lid < kSG; ++lid) {
                        const size_t row = h * kSG + lid;
                        const size_t blk = row * nbpr + bid;
                        const uint8_t* blk_src = src + blk * bb;

                        uint8_t pqs[128];
                        uint8_t pqh[32];
                        uint8_t psl[16];
                        shuffle_q5k_block(blk_src, pqs, pqh, psl);

                        sg_scatter_chunks(dst + pqs_entry, pqs, 128, lid);
                        sg_scatter_chunks(dst + pqh_entry, pqh, 32, lid);
                        // psl SoA fields: sl(4) ml(4) sh(2) mh(2) d(2) dmin(2).
                        uint8_t* pe = dst + psl_entry;
                        std::memcpy(pe + 0 + lid * 4, psl + 0, 4);    // sl
                        std::memcpy(pe + 64 + lid * 4, psl + 4, 4);   // ml
                        std::memcpy(pe + 128 + lid * 2, psl + 8, 2);  // sh
                        std::memcpy(pe + 160 + lid * 2, psl + 10, 2); // mh
                        std::memcpy(pe + 192 + lid * 2, psl + 12, 2); // d
                        std::memcpy(pe + 224 + lid * 2, psl + 14, 2); // dmin
                    }
                }
            });
        } else if (is_q6k) {  // Q6_K
            const size_t off_pql = 0;
            const size_t off_pqh = N * nbpr * 128;
            const size_t off_ps = off_pqh + N * nbpr * 64;
            const size_t off_pd = off_ps + N * nbpr * 16;
            ov::parallel_for(nrg, [&](size_t h) {
                for (size_t bid = 0; bid < nbpr; ++bid) {
                    const size_t pql_entry = off_pql + (h * nbpr + bid) * kSG * 128;
                    const size_t pqh_entry = off_pqh + (h * nbpr + bid) * kSG * 64;
                    const size_t ps_entry = off_ps + (h * nbpr + bid) * kSG * 16;
                    const size_t pd_entry = off_pd + (h * nbpr + bid) * kSG * 2;
                    for (size_t lid = 0; lid < kSG; ++lid) {
                        const size_t row = h * kSG + lid;
                        const size_t blk = row * nbpr + bid;
                        const uint8_t* blk_src = src + blk * bb;

                        uint8_t pql[128];
                        uint8_t pqh[64];
                        uint8_t ps[16];
                        uint8_t pd[2];
                        shuffle_q6k_block(blk_src, pql, pqh, ps, pd);

                        sg_scatter_chunks(dst + pql_entry, pql, 128, lid);
                        sg_scatter_chunks(dst + pqh_entry, pqh, 64, lid);
                        // ps: scale si at si*kSG + lid (1 byte/lane).
                        uint8_t* pse = dst + ps_entry;
                        for (size_t si = 0; si < 16; ++si) {
                            pse[si * kSG + lid] = ps[si];
                        }
                        // pd: 1 half (2 bytes) per lane.
                        std::memcpy(dst + pd_entry + lid * 2, pd, 2);
                    }
                }
            });
        } else {  // small-block: Q4_0 / Q4_1 / Q8_0 (8 native 32-elem blocks -> 256-elem super-block)
            // pqs: 128 B (4-bit) or 256 B (8-bit) chunk-interleaved; pd (+ pm for Q4_1): SoA fp16.
            const size_t pqs_bytes  = is_q8_0 ? 256 : 128;
            const size_t qs_off     = is_q4_1 ? 4 : 2;  // qs offset within native block (after d[/ m])
            const size_t off_pqs = 0;
            const size_t off_pd  = N * nbpr * pqs_bytes;
            const size_t off_pm  = off_pd + N * nbpr * 16;  // Q4_1 only
            ov::parallel_for(nrg, [&](size_t h) {
                for (size_t bid = 0; bid < nbpr; ++bid) {
                    const size_t pqs_entry = off_pqs + (h * nbpr + bid) * kSG * pqs_bytes;
                    const size_t pd_entry  = off_pd + (h * nbpr + bid) * kSG * 16;
                    const size_t pm_entry  = off_pm + (h * nbpr + bid) * kSG * 16;
                    for (size_t lid = 0; lid < kSG; ++lid) {
                        const size_t row = h * kSG + lid;

                        uint8_t pqs[256];
                        uint8_t pd[16];
                        uint8_t pm[16];
                        // Gather the 8 native blocks of this super-block (bid*8 .. bid*8+7).
                        for (size_t j = 0; j < 8; ++j) {
                            const size_t nb = bid * 8 + j;
                            const uint8_t* blk_src = src + (row * nbpr_native + nb) * bb;
                            // d (2 B) is the first field of every small-block format.
                            pd[j * 2 + 0] = blk_src[0];
                            pd[j * 2 + 1] = blk_src[1];
                            if (is_q4_1) {
                                pm[j * 2 + 0] = blk_src[2];
                                pm[j * 2 + 1] = blk_src[3];
                            }
                            const uint8_t* qs = blk_src + qs_off;
                            if (is_q8_0) {
                                for (size_t i = 0; i < 32; ++i) {
                                    pqs[j * 32 + i] = qs[i];  // 32 int8 weights (sub-block j)
                                }
                            } else {
                                for (size_t k = 0; k < 16; ++k) {
                                    pqs[j * 16 + k] = qs[k];  // qs[k] low@k / high@16+k -> sub-block j
                                }
                            }
                        }
                        sg_scatter_chunks(dst + pqs_entry, pqs, pqs_bytes, lid);
                        // pd / pm SoA: field j (0..7) fp16 at j*kSG*2 + lid*2.
                        for (size_t j = 0; j < 8; ++j) {
                            std::memcpy(dst + pd_entry + j * kSG * 2 + lid * 2, pd + j * 2, 2);
                        }
                        if (is_q4_1) {
                            for (size_t j = 0; j < 8; ++j) {
                                std::memcpy(dst + pm_entry + j * kSG * 2 + lid * 2, pm + j * 2, 2);
                            }
                        }
                    }
                }
            });
        }

        // New Constant: SAME element type, SAME shape, SAME byte size — only the bytes are shuffled.
        auto packed_const = std::make_shared<ov::op::v0::Constant>(wt, shape, dst_buf);
        packed_const->set_friendly_name(w_const->get_friendly_name());
        ov::copy_runtime_info(w_const, packed_const);
        // Rewire only this FC's weight input; each match shuffles its own input (idempotent: the gate is
        // per-node and the byte size never changes).
        fc->input(1).replace_source_output(packed_const->output(0));
        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(fc_m, "RepackGGUFWeightsShuffle");
    this->register_matcher(m, callback);
}

}  // namespace ov::intel_gpu
