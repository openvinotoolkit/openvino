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

class AutoTileConfiger {
public:
    AutoTileConfiger() {}
    ~AutoTileConfiger() {
        do_config(nullptr);
    }
    void do_config(void* cfg) {
        static ov::Extensions::Cpu::TileConfiger configer;
        if (cfg != last_cfg) {
            configer(cfg);
            last_cfg = cfg;
        }
    }

private:
    void* last_cfg = nullptr;
};

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


struct Work {
    std::vector<PlainTensor> weights;  // ov::bfloat16 weights for current thread

    std::shared_ptr<std::atomic_int> sync_flag;
    int n0 = 0;
    int n1 = 0;
    int k0 = 0;
    int k1 = 0;
    int BN = 0;
    int blk_K_size = 0;
    int output_id;
    ov::bfloat16* p_raw_weights;
    operator bool() {
        return BN > 0;
    }

    MKernel& get_MKernel() {
        constexpr int BM = 256;
        static MKernel jit_amx0(BM);
        return jit_amx0;
    }

    // input : weight [N, K], setup repacks range of N [n_start, n_end)
    template <typename T>
    void setup(T* p_weight, int stride) {
        auto& mkernel = get_MKernel();
        auto num_blk_K = (k1 - k0) / blk_K_size;
        auto* pw = p_weight + n0 * stride / sizeof(T) + k0;

        weights.resize(num_blk_K);
        for (int k = 0; k < num_blk_K; k++) {
            mkernel.prepareB(weights[k], pw + k * blk_K_size, stride, BN, blk_K_size);
        }

        for (int Mtails = 0; Mtails < 32; Mtails++) {
            mkernel.tile_config_M(m_tcfg[Mtails], Mtails == 0 ? 32 : Mtails);
        }
    }

    ov::Extensions::Cpu::TileConfig m_tcfg[32];
    AutoTileConfiger m_tile_configer;

    size_t get_C_size(int M) {
        auto Mtails = M % 32;
        auto Mbody = M - Mtails;
        auto C_M = Mbody + (Mtails ? 32 : 0);
        return C_M * BN;
    }

    void run(int M, uint8_t* pA, int strideA, PlainTensor& C) {
        auto& mkernel = get_MKernel();

        int num_blk_K = (k1 - k0) / blk_K_size;

        auto Mtails = M % 32;
        auto Mbody = M - Mtails;

        auto C_M = Mbody + (Mtails ? 32 : 0);
        C.resize<float>({static_cast<size_t>(C_M), static_cast<size_t>(BN)});
        auto pC = reinterpret_cast<uint8_t*>(C.ptr_v());

        pA += k0 * sizeof(ov::bfloat16);
        bool do_accumulation = false;

        for (int ki = 0; ki < num_blk_K; ki++) {
            PlainTensor& blockB = weights[ki];
            PlainTensor& blockB1 = weights[(ki + 1) < num_blk_K ? (ki + 1) : ki];
            if (Mbody) {
                m_tile_configer.do_config(&m_tcfg[0]);
                mkernel.run(Mbody,
                            pA + ki * blk_K_size * sizeof(ov::bfloat16),
                            strideA,
                            blockB,
                            pC,
                            C.stride_bytes(0),
                            reinterpret_cast<uint8_t*>(blockB1.ptr_v()),
                            do_accumulation);
            }

            if (Mtails) {
                m_tile_configer.do_config(&m_tcfg[Mtails]);
                mkernel.run(Mtails,
                            pA + ki * blk_K_size * sizeof(ov::bfloat16) + Mbody * strideA,
                            strideA,
                            blockB,
                            pC + Mbody * C.stride_bytes(0),
                            C.stride_bytes(0),
                            reinterpret_cast<uint8_t*>(blockB1.ptr_v()),
                            do_accumulation);
            }
            do_accumulation = true;
        }
        m_tile_configer.do_config(nullptr);
    }
};

// combine gate_proj & up_proj using activation algo, then convert to bf16
//     ConvertFP32toBF16(act_fn(gate) * up)
class GateUpCombine : public dnnl::impl::cpu::x64::jit_generator {
public:
    DECLARE_CPU_JIT_AUX_FUNCTIONS(GateUpCombine)

    const dnnl_alg_kind_t m_act_alg;
    GateUpCombine(dnnl_alg_kind_t act_alg) : jit_generator(jit_name()), m_act_alg(act_alg) {
        create_kernel();
    }

    void generate() override;

    void call(float* src, size_t src_stride, ov::bfloat16 * dst, size_t dst_stride, int num_rows, int num_cols) {
        for (int m = 0; m < num_rows; m++, src += src_stride, dst += dst_stride) {
            auto* prefetch_dst = (m + 1 < num_rows) ? (dst + dst_stride) : (dst);

            // gate_proj & up_proj are interleaved in unit of 16 elements into gateup
            //
            // for(int i = 0; i < N; i += 32) {
            //   for(int k = 0; k < 16; k++)
            //     gate = src[i+k]
            //     up = src[i+k+16]
            //     *dst++ = ConvertFP32toBF16(act_fn(gate) * up_proj)
            //   }
            // }
            //
            (*this)(src, dst, prefetch_dst, num_cols);
        }
    }
};

class ReduceAdd2bh : public dnnl::impl::cpu::x64::jit_generator {
public:
    DECLARE_CPU_JIT_AUX_FUNCTIONS(ReduceAdd2bh)

    bool m_do_reduce2;
    ReduceAdd2bh(bool do_reduce2) : jit_generator(jit_name()), m_do_reduce2(do_reduce2) {
        create_kernel();
    }

    void generate() override;

    // add two float input eltwise and convert to bf16 : ConvertFP32toBF16(src0 + src1)
    void call(float * src0, float * src1, size_t src_stride, ov::bfloat16 * dst, size_t dst_stride, int num_rows, int num_cols) {
        for (int m = 0; m < num_rows; m++, src0 += src_stride, src1 += src_stride, dst += dst_stride) {
            // the prefetch distance is increased to ensure by the time store happens
            // prefetch has done and no HW prefetcher is triggered
            auto* prefetch_dst = (m + 2 < num_rows) ? (dst + 2 * dst_stride) : (dst);
            (*this)(src0, src1, dst, prefetch_dst, num_cols);
        }
    }

    // convert tensor to bf16: ConvertFP32toBF16(src0)
    void call(float * src0, size_t src_stride, ov::bfloat16 * dst, size_t dst_stride, int num_rows, int num_cols) {
        for (int m = 0; m < num_rows; m++, src0 += src_stride, dst += dst_stride) {
            // the prefetch distance is increased to ensure by the time store happens
            // prefetch has done and no HW prefetcher is triggered
            auto* prefetch_dst = (m + 2 < num_rows) ? (dst + 2 * dst_stride) : (dst);
            (*this)(src0, dst, prefetch_dst, num_cols);
        }
    }
};

}  // namespace intel_cpu
}  // namespace ov
