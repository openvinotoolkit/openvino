// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "cpu/x64/cpu_isa_traits.hpp"
#include "cpu/x64/jit_generator.hpp"
#include "../scaled_attn/executor_pa_common.hpp"
#include "utils/plain_tensor.hpp"

namespace ov {
namespace intel_cpu {

class MKernel : public dnnl::impl::cpu::x64::jit_generator {
public:
    DECLARE_CPU_JIT_AUX_FUNCTIONS(MKernel)

    int m_prefetch_Blines;

    MKernel(int M_hint = 256) : jit_generator("MKernel") {
        setup(M_hint);
    }

    void generate() override;

    //  M_hint is only a hint for prefetching, set to 0 to avoid prefetch
    void setup(int M_hint = 0) {
        if (M_hint == 0) {
            m_prefetch_Blines = 0;
        } else {
            m_prefetch_Blines = 32768 * sizeof(ov::bfloat16) / 64 / M_hint;
        }

        create_kernel();
    }

    // M can change w/o code-regeneration
    // with the help of :
    //  - m_BM_hint controls dynamic behaviour of the kernel
    //  - tile config controls behaviour of tileload & TMUL
    void tile_config_M(ov::Extensions::Cpu::TileConfig& tile_cfg, int M);

    // row data is in layout [N, K], maybe smaller than [32, 16]
    template <typename T>
    void repackB(ov::bfloat16* dst, T* src, int N_stride, int N, int K);

    // weight is supposed to be of shape[N, K], stride in unit of bytes
    template <typename T>
    void prepareB(PlainTensor& ret, T* p_weight, int stride, int N, int K);

    // to save push/pop: do not use `abi_save_gpr_regs`
    uint8_t* prefetch_next_A_addr;

    struct call_args {
        const uint8_t* pA;  // bfloat16
        int64_t strideA;    // in bytes
        const uint8_t* pB;  // bfloat16
        const uint8_t* pC;  // float32
        int64_t strideC;    // in bytes
        const uint8_t* prefetch;
        int64_t k_tiles;  // K / 32
        int64_t do_accumulation;
        int64_t M;
    };

    // run L2 cache blocking kernel with size:
    //    [BM, BK]*[BK, BN] => [BM, BN]
    //
    // prefetch of A can be done inside of this level of kernel
    // since it's done in unit of 32-rows
    // but prefetch of next B must be specified by caller.
    //
    void run(int M,  // actual M
             uint8_t* pA,
             int strideA,              // A [M, K]
             PlainTensor& repacked_B,  // B [N/32, K*32] ov::bfloat16
             uint8_t* pC,
             int strideC,          // C [M, N]
             uint8_t* prefetch_B,  // prefetch B
             bool do_accumulation);
};

class GateUpCombine : public dnnl::impl::cpu::x64::jit_generator {
public:
    DECLARE_CPU_JIT_AUX_FUNCTIONS(GateUpCombine)

    const dnnl_alg_kind_t m_act_alg;
    GateUpCombine(dnnl_alg_kind_t act_alg) : jit_generator(jit_name()), m_act_alg(act_alg) {
        create_kernel();
    }

    void generate() override;
};

class ReduceAdd2bh : public dnnl::impl::cpu::x64::jit_generator {
public:
    DECLARE_CPU_JIT_AUX_FUNCTIONS(ReduceAdd2bh)

    bool m_do_reduce2;
    ReduceAdd2bh(bool do_reduce2) : jit_generator(jit_name()), m_do_reduce2(do_reduce2) {
        create_kernel();
    }

    void generate() override;
};

}  // namespace intel_cpu
}  // namespace ov
