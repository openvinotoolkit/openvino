// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "cpu/x64/cpu_isa_traits.hpp"
#include "cpu/x64/jit_generator.hpp"
#include "../scaled_attn/executor_pa_common.hpp"
#include "utils/plain_tensor.hpp"

// register blocking size for K dimension (1x2 AMX B-tiles)
#define REG_BLK_K_SIZE 32

// register blocking size for N dimension (1x2 AMX B-tiles)
#define REG_BLK_N_SIZE 32

// cache blocking sie for K dimension
#define CACHE_BLK_K_SIZE 256

// cache blocking sie for M dimension
#define CACHE_BLK_M_SIZE 256

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
            // next block size: 32 * N * sizeof(ov::bfloat16),
            // call number: N / 32 * M / 32
            // each call needs fetch: 32 * N * sizeof(ov::bfloat16) / (N / 32 * M / 32) = 32 * 1024 * sizeof(ov::bfloat16) / M
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
    void prepareB(PlainTensor& ret, ov::bfloat16* dst, T* p_weight, int stride, int N, int K);

    // interleaving two weights into one in unit of 16-column
    template <typename T>
    void prepareB(PlainTensor& ret, ov::bfloat16* dst, T* p_weight1, T* p_weight2, int stride, int N, int K);

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

class MKernel_1x2 : public dnnl::impl::cpu::x64::jit_generator {
public:
    DECLARE_CPU_JIT_AUX_FUNCTIONS(MKernel_1x2)

    using call_args = MKernel::call_args;

    MKernel_1x2() : jit_generator("MKernel_1x2") {
        create_kernel();
    }

    void generate() override;
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
    ov::float16* p_raw_weights;
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
    void setup(ov::bfloat16* dst, T* p_weight, int stride) {
        auto& mkernel = get_MKernel();
        auto num_blk_K = (k1 - k0 + blk_K_size - 1) / blk_K_size;
        auto* pw = p_weight + n0 * stride / sizeof(T);

        // weight is divided along K dimension into equal size blk_K_size, except last block.
        weights.resize(num_blk_K);
        for (int k = k0, ki = 0; k < k1;) {
            auto subK = std::min(blk_K_size, k1 - k);
            mkernel.prepareB(weights[ki], dst, pw + k, stride, BN, subK);
            dst += BN*subK;
            k += subK;
            ki++;
        }

        for (int Mtails = 0; Mtails < 32; Mtails++) {
            mkernel.tile_config_M(m_tcfg[Mtails], Mtails == 0 ? 32 : Mtails);
        }
    }

    template <typename T>
    void setup(ov::bfloat16* dst, T* p_weight1, T* p_weight2, int stride) {
        auto& mkernel = get_MKernel();
        auto num_blk_K = (k1 - k0 + blk_K_size - 1) / blk_K_size;
        auto* pw1 = p_weight1 + (n0/2) * stride / sizeof(T);
        auto* pw2 = p_weight2 + (n0/2) * stride / sizeof(T);

        // weight is divided along K dimension into equal size blk_K_size, except last block.
        weights.resize(num_blk_K);
        for (int k = k0, ki = 0; k < k1;) {
            auto subK = std::min(blk_K_size, k1 - k);
            mkernel.prepareB(weights[ki], dst, pw1 + k, pw2 + k, stride, BN, subK);
            dst += BN*subK;
            k += subK;
            ki++;
        }

        for (int Mtails = 0; Mtails < 32; Mtails++) {
            mkernel.tile_config_M(m_tcfg[Mtails], Mtails == 0 ? 32 : Mtails);
        }
    }

    ov::Extensions::Cpu::TileConfig m_tcfg[32];
    AutoTileConfiger m_tile_configer;

    PlainTensor m_C;

    size_t set_C(int M, float * ext_buff) {
        auto Mtails = M % 32;
        auto Mbody = M - Mtails;
        auto C_M = Mbody + (Mtails ? 32 : 0);
        m_C.resize<float>({static_cast<size_t>(C_M), static_cast<size_t>(BN)}, ext_buff);
        return C_M * BN * sizeof(float);
    }

    void run(int M, uint8_t* pA, int strideA) {
        auto& mkernel = get_MKernel();

        int num_blk_K = weights.size();

        auto Mtails = M % 32;
        auto Mbody = M - Mtails;
        auto C_M = Mbody + (Mtails ? 32 : 0);

        auto C_stride_bytes = BN * sizeof(float);
        OPENVINO_ASSERT(C_M * C_stride_bytes <= m_C.stride_bytes(0) * m_C.size(0));
        auto pC = reinterpret_cast<uint8_t*>(m_C.ptr_v());

        pA += k0 * sizeof(ov::bfloat16);

        if (M > 16 || num_blk_K ==1) {
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
                                C_stride_bytes,
                                reinterpret_cast<uint8_t*>(blockB1.ptr_v()),
                                do_accumulation);
                }

                if (Mtails) {
                    m_tile_configer.do_config(&m_tcfg[Mtails]);
                    mkernel.run(Mtails,
                                pA + ki * blk_K_size * sizeof(ov::bfloat16) + Mbody * strideA,
                                strideA,
                                blockB,
                                pC + Mbody * C_stride_bytes,
                                C_stride_bytes,
                                reinterpret_cast<uint8_t*>(blockB1.ptr_v()),
                                do_accumulation);
                }
                do_accumulation = true;
            }
        } else {
            static MKernel_1x2 jit;
            PlainTensor& blockB = weights[0];
            // number of blocks in N dimension (in unit of 32 columns)
            auto num_blkN = static_cast<int>(blockB.size(0));
            m_tile_configer.do_config(&m_tcfg[Mtails]);
            // original: bit0: 0-tilezero+skip load from mem, 1-tilezero+load from mem; tilestore
            // new: bit0: 0-skip load from mem, 1-load from mem; bit1: 0-skip tilezero, 1-tilezero; bit2: 0-skip store, 1-store
            // if M > 32, firstK: 1 1 0(store, tilezero, skip load),      the otherK except last: 1 0 1(store, skip tilezero, load) lastK: 1 0 1
            // else,      firstK: 0 1 0(skip store, tilezero, skip load), the otherK except last: 0 0 0(skip all),                  lastK: 1 0 0(store, skip tile zero, skip load)
            int do_accumulation;

            MKernel::call_args args;
            args.strideA = strideA;
            args.strideC = C_stride_bytes;
            args.M = Mtails;
            for (int ni = 0; ni < num_blkN; ni++) {
                args.pC = pC + ni * 32 * sizeof(float);
                do_accumulation = 0b010;
                for (int ki = 0; ki < num_blk_K; ki++) {
                    PlainTensor& blockB = weights[ki];
                    args.k_tiles = blockB.m_dims[1] / 32 / 32;
                    args.pA = pA + ki * blk_K_size * sizeof(ov::bfloat16);
                    args.pB = blockB.ptr<uint8_t>() + ni * blockB.stride_bytes(0);
                    args.do_accumulation = do_accumulation;
                    // prefetch next N block. In memory bound, it seems no prefetch will be better.
                    // args.prefetch = args.pB + (ni == num_blkN - 1 ? 0 : strideB);
                    // args.prefetch = args.pB;
                    // [M, K] * [K, N]: [1..32, 256] * [256, 32]
                    jit(&args);
                    do_accumulation = (ki == num_blk_K - 2) ? 0b100 : 0;
                }
            }
        }
        m_tile_configer.do_config(nullptr);
    }
};

// allocate weight memory in bigger trunck can benefit from HugePage (with much less page-fault effort)
struct WeightBuffer {
    PlainTensor buffer;
    std::vector<size_t> offsets;
    void alloc(std::vector<Work>& works) {
        size_t weight_cnt = 0;
        for (auto& work : works) {
            offsets.push_back(weight_cnt);
            weight_cnt += (work.n1 - work.n0) * (work.k1 - work.k0);
        }
        buffer.resize<ov::bfloat16>({weight_cnt});
    }
    ov::bfloat16* get(int work_id) {
        return buffer.ptr<ov::bfloat16>() + offsets[work_id];
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
