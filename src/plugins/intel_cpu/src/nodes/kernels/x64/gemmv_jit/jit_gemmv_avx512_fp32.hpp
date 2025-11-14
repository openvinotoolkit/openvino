// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "gemmv_ukernel.hpp"

#include <memory>
#include <vector>

// Xbyak from oneDNN third_party include
#include "xbyak/xbyak.h"

namespace ov::intel_cpu::x64::gemmv_jit {

// AVX-512 FP32 GEMM-v (int8/u8 -> fp32), M block = 16 lanes (zmm)
class JitGemmvAvx512Fp32 : public GemmvKernel, private Xbyak::CodeGenerator {
public:
    struct CallArgs {
        const float* x;           // A[K] fp32
        const uint8_t* wq;        // packed Wq block base (size K*M_blk)
        const float* scales;      // per-tensor: 1, per-channel: M_blk
        const int32_t* zps;       // nullable, per-tensor:1, per-channel:M_blk
        const float* bias;        // nullable, per-tensor:1 or per-channel:M_blk
        float* y;                 // output base for this M-block
        int K;                    // K length
        int M_tail;               // lanes valid in this block (0/full)
        int ld_w_bytes;           // stride to next M-block (bytes), unused here
        int gran;                 // quant_granularity_t
        int w_nbits;              // 8 or 4
        int w_unsigned;           // 1 if unsigned (u8/u4), 0 if signed (i8/i4)
        int k_step_bytes;         // bytes advanced per K-step inside a block (16 for 8-bit, 8 for 4-bit)
        int accumulate;           // accumulate mode
        float sum_x;              // optional sum of X for zp compensation (precomputed outside)
        int fuse_gate;            // 1: multiply output by gate
        float gate;               // gate scale
        int act_kind;             // 0: none, 1: ReLU
        // debug (set by caller when needed)
        int dbg_enable;
        int dbg_k;
        float* dbg_q;             // decoded q (fp32) for capture
        float* dbg_qs;            // decoded q*s (fp32)

        // VNNI no-repack path (optional)
        // When use_vnni!=0 and CPU supports AVX512_VNNI and w_nbits==8 and w_unsigned==0 (i8),
        // kernel will use u8(X) x s8(W) -> s32 accumulation with vpdpbusd, without repacking W.
        const uint8_t* x_q8 = nullptr; // quantized X[K] (u8), per-tensor
        float s_x = 1.f;                // per-tensor X scale (for dequant in epilogue)
        int32_t zp_x = 128;             // per-tensor X zero-point (usually 128)
        int32_t sum_x_q = 0;            // sum of x_q8 bytes over K (for zp_w compensation)
        int use_vnni = 0;               // enable VNNI path
        int need_sumw = 0;              // compute sumW (required when zp_x!=0 or zps provided)
    };

    JitGemmvAvx512Fp32();
    void operator()(const gemmv_ukr_params_t* p) const override;
    const char* name() const override { return "jit_avx512_fp32"; }

private:
    using fn_t = void(*)(const CallArgs*);
    fn_t fn_ = nullptr;
};

} // namespace ov::intel_cpu::x64::gemmv_jit
