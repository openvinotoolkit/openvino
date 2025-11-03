// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "gemmv_ukernel.hpp"
#include "xbyak/xbyak.h"

namespace ov::intel_cpu::x64::gemmv_jit {

// AVX-512 VNNI GEMV for u8 (X) x s8 (W) -> s32, per-tensor quant, M block = 16, K grouped by 4 bytes
class JitGemmvAvx512VnniS32 : private Xbyak::CodeGenerator {
public:
    struct CallArgs {
        const uint8_t* xq;     // [K]
        const uint8_t* wq;     // base for current M-block, K grouped by 4, ld_w_gbytes = (K/4)*64
        int K_groups;          // K/4 rounded up
        int ld_w_gbytes;       // bytes between successive M-blocks (groups * 64)
        float* y;              // output base for current M-block
        int M_tail;            // 0 or (1..15)

        // per-tensor quant params
        float s_w;
        float s_x;
        int32_t zp_w;          // may be 0
        int32_t zp_x;          // typically 128
        int32_t sum_x_q;       // sum of xq bytes over K (for zp_w compensation)
        float bias;            // optional bias per-tensor (else 0)

        // Optional debug capture (s32 accumulators):
        int dbg_enable;        // 1 to enable capture into dbg_*
        int32_t* dbg_acc;      // [16] s32 dot-product per lane
        int32_t* dbg_sumw;     // [16] s32 sum of weights per lane
        int dbg_dump_only;     // 1: capture and return early (skip epilogue/store)
    };

    JitGemmvAvx512VnniS32();
    using fn_t = void(*)(const CallArgs*);
    fn_t get() const { return fn_; }

private:
    fn_t fn_ = nullptr;
};

// Wrapper: runs VNNI kernel for per-tensor case; returns false if unsupported (no VNNI)
bool run_gemmv_vnni_i8u8_fp32(const uint8_t* xq, int K,
                              const uint8_t* wq_k4, int M, int ld_w_gbytes,
                              float s_w, int32_t zp_w, float s_x, int32_t zp_x,
                              float* y, float bias,
                              int dbg_block, int32_t* dbg_acc, int32_t* dbg_sumw,
                              int dbg_dump_only);

} // namespace ov::intel_cpu::x64::gemmv_jit
