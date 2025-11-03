// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "jit_gemmv_avx2_fp32.hpp"

#include <algorithm>
#include <cmath>

namespace ov::intel_cpu::x64::gemmv_jit {

static inline float get_scale(const float* s, int lane_m, quant_granularity_t gran, int base_m, int group_size) {
    if (gran == quant_granularity_t::per_tensor) return s[0];
    if (gran == quant_granularity_t::per_channel) return s[lane_m];
    // per_group
    const int gs = group_size > 0 ? group_size : 16;
    const int g = (base_m + lane_m) / gs;
    return s[g];
}
static inline int32_t get_zp(const int32_t* z, int lane_m, quant_granularity_t gran, int base_m, int group_size) {
    if (!z) return 0;
    if (gran == quant_granularity_t::per_tensor) return z[0];
    if (gran == quant_granularity_t::per_channel) return z[lane_m];
    const int gs = group_size > 0 ? group_size : 16;
    const int g = (base_m + lane_m) / gs;
    return z[g];
}
static inline float get_bias(const float* b, int lane_m, quant_granularity_t gran, int base_m, int group_size) {
    if (!b) return 0.f;
    if (gran == quant_granularity_t::per_tensor) return b[0];
    if (gran == quant_granularity_t::per_channel) return b[lane_m];
    const int gs = group_size > 0 ? group_size : 16;
    const int g = (base_m + lane_m) / gs;
    return b[g];
}

void RefGemmvFp32::operator()(const gemmv_ukr_params_t* p) const {
    const int M_blk = 16;
    const int M_full = p->M / M_blk;
    const int M_tail = p->M % M_blk;

    const float* x = static_cast<const float*>(p->x);
    auto* y = static_cast<float*>(p->y);

    float sumx = 0.f;
    if (p->zps) {
        for (int k = 0; k < p->K; ++k) sumx += x[k];
    }

    auto do_block = [&](int block_idx, int valid) {
        const uint8_t* wq = p->wq + block_idx * p->ld_w_bytes;
        const int base_m = p->m_base + block_idx * M_blk;
        // For per_channel, advance metadata pointers to the start of this M-block;
        // per_tensor/per_group are interpreted inside helpers via base_m.
        const float* sc = (p->gran == quant_granularity_t::per_channel)
                              ? (p->scales ? (p->scales + base_m) : nullptr)
                              : p->scales;
        const int32_t* zp = (p->gran == quant_granularity_t::per_channel)
                                ? (p->zps ? (p->zps + base_m) : nullptr)
                                : p->zps;
        const float* bs = (p->gran == quant_granularity_t::per_channel)
                              ? (p->bias ? (p->bias + base_m) : nullptr)
                              : p->bias;
        float* yb = y + block_idx * M_blk;

        float acc[M_blk] = {0};
        if (p->accumulate) {
            for (int m = 0; m < valid; ++m) acc[m] = yb[m];
        }

        if (p->w_type == w_dtype_t::i4 || p->w_type == w_dtype_t::u4) {
            for (int k = 0; k < p->K; ++k) {
                const float xk = x[k];
                const uint8_t* bp = wq + k * (M_blk / 2);
                for (int m = 0; m < valid; ++m) {
                    const int idx = m >> 1;
                    const uint8_t b = bp[idx];
                    uint8_t nib = (m & 1) ? ((b >> 4) & 0x0F) : (b & 0x0F);
                    int32_t q = (p->w_type == w_dtype_t::u4) ? (int32_t)nib : (int32_t)((nib ^ 0x8) - 0x8);
                    const float s = get_scale(sc, m, p->gran, base_m, p->group_size);
                    const float wr = static_cast<float>(q) * s;
                    acc[m] += wr * xk;
                }
            }
        } else {
            for (int k = 0; k < p->K; ++k) {
                const float xk = x[k];
                const uint8_t* wk = wq + k * M_blk;
                for (int m = 0; m < valid; ++m) {
                    int32_t q = static_cast<int32_t>(wk[m]);
                    if (p->w_type == w_dtype_t::i8) {
                        // sign extend
                        q = static_cast<int8_t>(wk[m]);
                    }
                    const float s = get_scale(sc, m, p->gran, base_m, p->group_size);
                    const float wr = static_cast<float>(q) * s;
                    acc[m] += wr * xk;
                }
            }
        }

        // zp compensation + bias
        for (int m = 0; m < valid; ++m) {
            const float s = get_scale(sc, m, p->gran, base_m, p->group_size);
            const float b = get_bias(bs, m, p->gran, base_m, p->group_size);
            const float z = static_cast<float>(get_zp(zp, m, p->gran, base_m, p->group_size));
            acc[m] += b - s * z * sumx;
        }

        // Optional MoE epilogue: gate then activation (ReLU)
        if (p->fuse_gate && p->gate_scale != 1.f) {
            for (int m = 0; m < valid; ++m) acc[m] *= p->gate_scale;
        }
        if (p->act_kind == 1) {
            for (int m = 0; m < valid; ++m) acc[m] = acc[m] < 0.f ? 0.f : acc[m];
        } else if (p->act_kind == 2) {
            for (int m = 0; m < valid; ++m) {
                const float v = acc[m];
                acc[m] = v / (1.f + std::exp(-v)); // SiLU = x * sigmoid(x)
            }
        }

        for (int m = 0; m < valid; ++m) yb[m] = acc[m];
    };

    for (int bi = 0; bi < M_full; ++bi) do_block(bi, M_blk);
    if (M_tail) do_block(M_full, M_tail);
}

} // namespace ov::intel_cpu::x64::gemmv_jit
