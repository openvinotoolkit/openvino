// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "../scaled_attn/executor_pa_common.hpp"
#include "cpu/x64/cpu_isa_traits.hpp"
#include "cpu/x64/jit_generator.hpp"
#include "utils/plain_tensor.hpp"

// register blocking size for K dimension (1x2 AMX B-tiles)
constexpr int REG_BLK_K_SIZE = 32;
constexpr int REG_BLK_K_SIZE_I8 = 64;

// register blocking size for N dimension (1x2 AMX B-tiles)
constexpr int REG_BLK_N_SIZE = 32;

// cache blocking sie for K dimension
constexpr int CACHE_BLK_K_SIZE = 256;

// cache blocking sie for M dimension
constexpr int CACHE_BLK_M_SIZE = 256;

namespace ov::intel_cpu {

class AutoTileConfiger {
public:
    AutoTileConfiger() = default;
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

enum class TMUL_TYPE { SSD = 1, USD = 2, SUD = 3, UUD = 4, FP16 = 5, BF16 = 6 };

class MKernel : public dnnl::impl::cpu::x64::jit_generator {
public:
    DECLARE_CPU_JIT_AUX_FUNCTIONS(MKernel)

    int m_prefetch_Blines;
    const TMUL_TYPE m_tmul_type;
    int m_tile_reg_ksize;
    int m_M_hint;

    MKernel(int M_hint, TMUL_TYPE tmul_type) : jit_generator("MKernel"), m_tmul_type(tmul_type), m_M_hint(M_hint) {
        if (m_tmul_type == TMUL_TYPE::FP16 || m_tmul_type == TMUL_TYPE::BF16) {
            m_tile_reg_ksize = 32;
        } else {
            m_tile_reg_ksize = 64;
        }
        setup(M_hint);
    }

    void tmul(const Xbyak::Tmm& x1, const Xbyak::Tmm& x2, const Xbyak::Tmm& x3) {
        switch (m_tmul_type) {
        case TMUL_TYPE::SSD:
            tdpbssd(x1, x2, x3);
            break;
        case TMUL_TYPE::USD:
            tdpbusd(x1, x2, x3);
            break;
        case TMUL_TYPE::SUD:
            tdpbsud(x1, x2, x3);
            break;
        case TMUL_TYPE::UUD:
            tdpbuud(x1, x2, x3);
            break;
        case TMUL_TYPE::FP16:
            tdpfp16ps(x1, x2, x3);
            break;
        case TMUL_TYPE::BF16:
            tdpbf16ps(x1, x2, x3);
            break;
        }
    }

    void generate() override {
        if (m_M_hint <= 16) {
            generate_1x2();
        } else {
            generate_2x2();
        }
    }

    void generate_2x2();
    void generate_1x2();

    //  M_hint is only a hint for prefetching, set to 0 to avoid prefetch
    void setup(int M_hint = 0) {
        if (M_hint == 0) {
            m_prefetch_Blines = 0;
        } else {
            // next block size: 32 * N * sizeof(ov::bfloat16),
            // call number: N / 32 * M / 32
            // each call needs fetch: 32 * N * sizeof(ov::bfloat16) / (N / 32 * M / 32) = 32 * 1024 *
            // sizeof(ov::bfloat16) / M
            m_prefetch_Blines = 32768 * sizeof(ov::bfloat16) / 64 / M_hint;
        }

        create_kernel();
    }

    // M can change w/o code-regeneration
    // with the help of :
    //  - m_BM_hint controls dynamic behaviour of the kernel
    //  - tile config controls behaviour of tileload & TMUL
    void tile_config_M(ov::Extensions::Cpu::TileConfig& tile_cfg, int M);

    // to save push/pop: do not use `abi_save_gpr_regs`
    uint8_t* prefetch_next_A_addr;

    struct call_args {
        const uint8_t* pA;  // bfloat16/int8
        int64_t strideA;    // in bytes
        const uint8_t* pB;  // bfloat16/int8
        const uint8_t* pC;  // float32/int32
        int64_t strideC;    // in bytes
        const uint8_t* prefetch;
        int64_t k_tiles;  // K / 32
        int64_t do_accumulation;
        int64_t M;
    };

    // each 32x16(64x16) sub-matrix in [K, N]-shaped BMatrix to be loaded as B-tile is packed into AMX B-tile layout
    // and two neighboring B-tiles in same row are grouped as a pair (B0-B1), and all such pairs are arranged in [nN,
    // nK] shape
    struct BMatrix {
        uint8_t* ptr;
        // Bpair is two 1KB sub-matrixes repacked in AMX-Btile layout
        const size_t Bpair_size = 2048;
        size_t Bpair_rows;
        size_t Bpair_cols;

        // convert
        template <typename Tdst>
        void setup(Tdst* ext_buff, ov::float16* p_weight, int stride, int N, int K);

        void setup(int8_t* ext_buff, int8_t* p_weight, int stride, int N, int K);
        // two B tiles in each pair (B0 & B1) comes from different raw weight matrix
        template <typename Tdst>
        void setup(Tdst* ext_buff, ov::float16* p_weight_B0, ov::float16* p_weight_B1, int stride, int N, int K);

        void setup(int8_t* ext_buff, int8_t* p_weight_B0, int8_t* p_weight_B1, int stride, int N, int K);
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
             int strideA,          // A [M, K]
             BMatrix& repacked_B,  // B
             uint8_t* pC,
             int strideC,          // C [M, N]
             uint8_t* prefetch_B,  // prefetch B
             bool do_accumulation);
};

struct Work {
    std::vector<MKernel::BMatrix> weights;

    // will be used only when activation is being quantized asymmetrically
    PlainTensor w_sum_per_oc;

    std::shared_ptr<std::atomic_int> sync_flag;
    int n0 = 0;
    int n1 = 0;
    int k0 = 0;
    int k1 = 0;
    int BN = 0;
    int blk_K_size = 0;
    int output_id;
    void* p_raw_weights;
    operator bool() {
        return BN > 0;
    }

    bool quant_i8 = false;
    bool is_f16 = false;

    MKernel& get_MKernel() {
        constexpr int BM = 256;
        static MKernel jit_amx_bf16(BM, TMUL_TYPE::BF16);
        static MKernel jit_amx_f16(BM, TMUL_TYPE::FP16);
        static MKernel jit_amx_i8(BM, TMUL_TYPE::SSD);
        if (quant_i8) {
            return jit_amx_i8;
        }
        if (is_f16) {
            return jit_amx_f16;
        }
        return jit_amx_bf16;
    }

    MKernel& get_MKernel_1x2() {
        static MKernel jit_amx_bf16(16, TMUL_TYPE::BF16);
        static MKernel jit_amx_f16(16, TMUL_TYPE::FP16);
        static MKernel jit_amx_i8(16, TMUL_TYPE::SSD);
        if (quant_i8) {
            return jit_amx_i8;
        }
        if (is_f16) {
            return jit_amx_f16;
        }
        return jit_amx_bf16;
    }

    // input : weight [N, K], setup repacks range of N [n_start, n_end)
    template <typename Tsrc, typename Tdst>
    void setup(Tdst* dst, Tsrc* p_weight, int stride_in_bytes, bool do_sum_per_oc = false) {
        auto& mkernel = get_MKernel();
        auto num_blk_K = (k1 - k0 + blk_K_size - 1) / blk_K_size;
        auto* pw = p_weight + n0 * stride_in_bytes / sizeof(Tsrc);

        if (do_sum_per_oc) {
            w_sum_per_oc.resize<float>({static_cast<size_t>(n1 - n0)});
            auto* p_wsum_per_oc = w_sum_per_oc.ptr<float>();
            auto* pw_temp = pw;
            for (int n = n0; n < n1; n++, pw_temp += stride_in_bytes / sizeof(Tsrc)) {
                float fsum = 0;
                for (int k = k0; k < k1; k++) {
                    fsum += pw_temp[k];
                }
                *p_wsum_per_oc++ = fsum;
            }
        }

        // weight is divided along K dimension into equal size blk_K_size, except last block.
        weights.resize(num_blk_K);
        for (int k = k0, ki = 0; k < k1;) {
            auto subK = std::min(blk_K_size, k1 - k);
            weights[ki].setup(dst, pw + k, stride_in_bytes, BN, subK);
            dst += BN * subK;
            k += subK;
            ki++;
        }

        for (int Mtails = 0; Mtails < 32; Mtails++) {
            mkernel.tile_config_M(m_tcfg[Mtails], Mtails == 0 ? 32 : Mtails);
        }
    }

    // two weight matrix interleaved in unit of B-tiles
    // in each Bpair, p_weight1 stored in B0, p_weight2 stored in B1
    template <typename Tsrc, typename Tdst>
    void setup(Tdst* dst, Tsrc* p_weight1, Tsrc* p_weight2, int stride_in_bytes, bool do_sum_per_oc = false) {
        auto& mkernel = get_MKernel();
        auto num_blk_K = (k1 - k0 + blk_K_size - 1) / blk_K_size;
        auto* pw1 = p_weight1 + (n0 / 2) * stride_in_bytes / sizeof(Tsrc);
        auto* pw2 = p_weight2 + (n0 / 2) * stride_in_bytes / sizeof(Tsrc);

        if (do_sum_per_oc) {
            w_sum_per_oc.resize<float>({static_cast<size_t>(n1 - n0)});
            auto* p_wsum_per_oc = w_sum_per_oc.ptr<float>();
            auto* pw1_temp = pw1;
            auto* pw2_temp = pw2;
            auto stride_temp = stride_in_bytes / sizeof(Tsrc);
            for (int n = n0; n < n1; n += 32) {
                for (int dn = 0; dn < 16; dn++, pw1_temp += stride_temp) {
                    float fsum = 0;
                    for (int k = k0; k < k1; k++) {
                        fsum += pw1_temp[k];
                    }
                    *p_wsum_per_oc++ = fsum;
                }
                for (int dn = 0; dn < 16; dn++, pw2_temp += stride_temp) {
                    float fsum = 0;
                    for (int k = k0; k < k1; k++) {
                        fsum += pw2_temp[k];
                    }
                    *p_wsum_per_oc++ = fsum;
                }
            }
        }

        // weight is divided along K dimension into equal size blk_K_size, except last block.
        weights.resize(num_blk_K);
        for (int k = k0, ki = 0; k < k1;) {
            auto subK = std::min(blk_K_size, k1 - k);
            weights[ki].setup(dst, pw1 + k, pw2 + k, stride_in_bytes, BN, subK);
            dst += BN * subK;
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

    size_t set_C(int M, float* ext_buff) {
        auto Mtails = M % 32;
        auto Mbody = M - Mtails;
        auto C_M = Mbody + (Mtails ? 32 : 0);
        m_C.resize<float>({static_cast<size_t>(C_M), static_cast<size_t>(BN)}, ext_buff);
        return C_M * BN * sizeof(float);
    }

    void run(int M, uint8_t* pA, int strideA) {
        auto& mkernel = get_MKernel();

        auto num_blk_K = weights.size();

        auto Mtails = M % 32;
        auto Mbody = M - Mtails;
        auto C_M = Mbody + (Mtails ? 32 : 0);

        auto C_stride_bytes = BN * sizeof(float);
        OPENVINO_ASSERT(C_M * C_stride_bytes <= m_C.stride_bytes(0) * m_C.size(0));
        auto pC = reinterpret_cast<uint8_t*>(m_C.ptr_v());

        auto element_size = quant_i8 ? sizeof(int8_t) : sizeof(ov::bfloat16);

        pA += k0 * element_size;

        if (M > 16 || num_blk_K == 1) {
            bool do_accumulation = false;
            for (size_t ki = 0; ki < num_blk_K; ki++) {
                auto& blockB = weights[ki];
                auto& blockB1 = weights[(ki + 1) < num_blk_K ? (ki + 1) : ki];
                if (Mbody) {
                    m_tile_configer.do_config(&m_tcfg[0]);
                    mkernel.run(Mbody,
                                pA + ki * blk_K_size * element_size,
                                strideA,
                                blockB,
                                pC,
                                C_stride_bytes,
                                blockB1.ptr,
                                do_accumulation);
                }

                if (Mtails) {
                    m_tile_configer.do_config(&m_tcfg[Mtails]);
                    mkernel.run(Mtails,
                                pA + ki * blk_K_size * element_size + Mbody * strideA,
                                strideA,
                                blockB,
                                pC + Mbody * C_stride_bytes,
                                C_stride_bytes,
                                blockB1.ptr,
                                do_accumulation);
                }
                do_accumulation = true;
            }
        } else {
            auto& jit = get_MKernel_1x2();
            auto& blockB = weights[0];
            // number of blocks in N dimension (in unit of 32 columns)
            auto num_blkN = blockB.Bpair_cols;
            m_tile_configer.do_config(&m_tcfg[Mtails]);
            // original: bit0: 0-tilezero+skip load from mem, 1-tilezero+load from mem; tilestore
            // new: bit0: 0-skip load from mem, 1-load from mem; bit1: 0-skip tilezero, 1-tilezero; bit2: 0-skip store,
            // 1-store if M > 32, firstK: 1 1 0(store, tilezero, skip load)
            //      the otherK except last: 1 0 1(store, skip tilezero, load) lastK: 1 0 1
            // else
            //      firstK: 0 1 0(skip store, tilezero, skip load), the otherK except last: 0 0 0(skip all),
            //      lastK: 1 0 0(store, skip tile zero, skip load)
            int do_accumulation;
            MKernel::call_args args;
            args.strideA = strideA;
            args.strideC = C_stride_bytes;
            args.M = Mtails;
            for (size_t ni = 0; ni < num_blkN; ni++) {
                args.pC = pC + ni * 32 * sizeof(float);
                do_accumulation = 0b010;
                for (size_t ki = 0; ki < num_blk_K; ki++) {
                    auto& blockB = weights[ki];
                    args.k_tiles = blockB.Bpair_rows;
                    args.pA = pA + ki * blk_K_size * element_size;
                    args.pB = blockB.ptr + ni * blockB.Bpair_rows * blockB.Bpair_size;
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
    void alloc(std::vector<Work>& works, int element_size) {
        size_t weight_size = 0;
        for (auto& work : works) {
            offsets.push_back(weight_size);
            weight_size += (work.n1 - work.n0) * (work.k1 - work.k0) * element_size;
        }
        buffer.resize<int8_t>({weight_size});
    }
    template <typename T = ov::bfloat16>
    T* get(int work_id) {
        return reinterpret_cast<T*>(buffer.ptr<int8_t>() + offsets[work_id]);
    }
};

struct ScratchBuffAllocator {
    using CallBack = std::function<void(void* ptr)>;
    std::vector<CallBack> m_allocs;
    std::vector<size_t> m_sizes;
    size_t m_total_size = 0;
    ScratchBuffAllocator() = default;

    // register size / allocate totally size / inform consumers
    void register_allocation(size_t size, const CallBack& cb) {
        m_allocs.push_back(cb);
        m_total_size += size;
        m_sizes.push_back(size);
    }
    size_t size() {
        return m_total_size;
    }
    void finalize(void* base) {
        auto* ptr = reinterpret_cast<uint8_t*>(base);
        for (size_t i = 0; i < m_allocs.size(); i++) {
            m_allocs[i](ptr);
            ptr += m_sizes[i];
        }
    }
};

struct MatrixDynQuantPerRow {
    // M x K
    int M;
    int K;
    int8_t* data;
    float* scale;
    float* zp;
    bool asym = true;

    MatrixDynQuantPerRow() = default;

    size_t size() {
        // size of data & scale & zp
        return M * K + M * sizeof(float) * 2;
    }

    size_t stride() {
        return K;
    }

    void setup(void* ext_buf) {
        data = reinterpret_cast<int8_t*>(ext_buf);
        scale = reinterpret_cast<float*>(data + M * K);
        zp = reinterpret_cast<float*>(scale + M);
    }

    void quantize(size_t BM, ov::bfloat16* psrc, int src_stride);
    void quantize(size_t BM, ov::float16* psrc, int src_stride);
};

// combine gate_proj & up_proj using activation algo, then convert to bf16
//     ConvertFP32toBF16(act_fn(gate) * up)
class GateUpCombine : public dnnl::impl::cpu::x64::jit_generator {
public:
    DECLARE_CPU_JIT_AUX_FUNCTIONS(GateUpCombine)

    const dnnl_alg_kind_t m_act_alg;
    const bool m_to_f16;

    GateUpCombine(dnnl_alg_kind_t act_alg, bool to_f16)
        : jit_generator(jit_name()),
          m_act_alg(act_alg),
          m_to_f16(to_f16) {
        create_kernel();
    }

    void generate() override;

    void call(float* src, size_t src_stride, void* pv_dst, size_t dst_stride, int num_rows, int num_cols) {
        auto* dst = reinterpret_cast<int16_t*>(pv_dst);
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

    const bool m_do_reduce2;
    const bool m_to_f16;
    ReduceAdd2bh(bool do_reduce2, bool to_f16) : jit_generator(jit_name()), m_do_reduce2(do_reduce2), m_to_f16(to_f16) {
        create_kernel();
    }

    void generate() override;

    struct CallArgs {
        float* src0;
        float* src1;
        int16_t* dst;
        int16_t* prefetch_dst;
        int64_t num_cols;
    };
    // add two float input eltwise and convert to bf16 : ConvertFP32toBF16(src0 + src1)
    void
    call(float* src0, float* src1, size_t src_stride, void* pf16_dst, size_t dst_stride, int num_rows, int num_cols) {
        CallArgs args;
        args.src0 = src0;
        args.src1 = src1;
        args.dst = reinterpret_cast<int16_t*>(pf16_dst);
        args.num_cols = num_cols;
        for (int m = 0; m < num_rows; m++, args.src0 += src_stride, args.src1 += src_stride, args.dst += dst_stride) {
            // the prefetch distance is increased to ensure by the time store happens
            // prefetch has done and no HW prefetcher is triggered
            args.prefetch_dst = (m + 2 < num_rows) ? (args.dst + 2 * dst_stride) : (args.dst);

            (*this)(&args);
        }
    }

    // convert tensor to bf16: ConvertFP32toBF16(src0)
    void call(float* src0, size_t src_stride, void* pf16_dst, size_t dst_stride, int num_rows, int num_cols) {
        CallArgs args;
        args.src0 = src0;
        args.dst = reinterpret_cast<int16_t*>(pf16_dst);
        args.num_cols = num_cols;
        for (int m = 0; m < num_rows; m++, args.src0 += src_stride, args.dst += dst_stride) {
            // the prefetch distance is increased to ensure by the time store happens
            // prefetch has done and no HW prefetcher is triggered
            args.prefetch_dst = (m + 2 < num_rows) ? (args.dst + 2 * dst_stride) : (args.dst);
            (*this)(&args);
        }
    }
};

}  // namespace ov::intel_cpu
