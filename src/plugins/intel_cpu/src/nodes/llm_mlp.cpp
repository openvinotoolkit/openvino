// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "llm_mlp.h"

#include <chrono>
#include <string>
#include <vector>

#include "common/bfloat16.hpp"
#include "common/cpu_memcpy.h"
#include "cpu/x64/cpu_isa_traits.hpp"
#include "cpu/x64/jit_generator.hpp"
#include "shape_inference/shape_inference_internal_dyn.hpp"
#include "utils/plain_tensor.hpp"

#if defined(OPENVINO_ARCH_X86_64)
#include "kernels/x64/mlp_kernel.hpp"
using TileConfig = ov::Extensions::Cpu::TileConfig;
using TileConfiger = ov::Extensions::Cpu::TileConfiger;
#endif

namespace ov {
namespace intel_cpu {
namespace node {

namespace {

#if defined(OPENVINO_ARCH_X86_64)
using namespace dnnl::impl::cpu::x64;

class AutoTileConfiger {
public:
    AutoTileConfiger() {}
    ~AutoTileConfiger() {
        do_config(nullptr);
    }
    void do_config(void* cfg) {
        static TileConfiger configer;
        if (cfg != last_cfg) {
            configer(cfg);
            last_cfg = cfg;
        }
    }

private:
    void* last_cfg = nullptr;
};

static PlainTensor& getC(int ithr) {
    static std::vector<PlainTensor> all_C(parallel_get_max_threads());
    return all_C[ithr];
}

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

    TileConfig m_tcfg[32];
    AutoTileConfiger m_tile_configer;

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

class Linear {
public:
    std::vector<Work> works;

    int used_nthr = 0;
    bool do_splitK = false;

    Linear() {}

    // weight [N, K]
    // Gate & Up are interleaved in N dimension: 16-gate / 16-up
    // and post-ops will compute  silu(gate)*up in unit of 16 elements
    // and store out as bfloat16.
    template <typename T>
    void setup(T* p_weight, int stride, int N, int K, bool _do_splitK = false) {
        const int blk_K_size = 256;
        // prepare weights, split N among threads
        // in unit of 32
        OPENVINO_ASSERT((N % 32) == 0);
        OPENVINO_ASSERT((K % blk_K_size) == 0);
        auto nthr = parallel_get_max_threads();
        auto num_blk_N = N / 32;
        auto num_blk_K = K / blk_K_size;
        works.resize(nthr);

        do_splitK = _do_splitK;
        auto K_splits = do_splitK ? 2 : 1;
        // split task on more cores is better on TBB
        auto valid_nthr = nthr / K_splits;
        auto blkN_per_thread = (num_blk_N) / valid_nthr;
        auto blkN_leftover = num_blk_N - (blkN_per_thread * valid_nthr);
        auto start_blkN = 0;
        used_nthr = 0;
        auto blkK_per_thread = (num_blk_K + K_splits - 1) / K_splits;

        for (int ithr = 0; ithr < nthr; ithr += K_splits) {
            auto blkN = std::min(num_blk_N - start_blkN, blkN_per_thread);
            if (blkN_leftover > 0) {
                blkN_leftover--;
                blkN++;
            }
            if (blkN) {
                auto shared_atomic = std::make_shared<std::atomic_int>(0);
                auto start_blkK = 0;
                for (int ik = 0; ik < K_splits; ik++) {
                    auto blk_K = std::min(num_blk_K - start_blkK, blkK_per_thread);

                    auto& work = works[ithr + ik];

                    work.sync_flag = shared_atomic;
                    work.blk_K_size = blk_K_size;

                    work.n0 = (start_blkN)*32;
                    work.n1 = (start_blkN + blkN) * 32;
                    work.BN = blkN * 32;
                    work.k0 = start_blkK * blk_K_size;
                    work.k1 = (start_blkK + blk_K) * blk_K_size;

                    start_blkK += blk_K;
                    used_nthr++;
                }
            }

            start_blkN += blkN;
        }

        DEBUG_LOG("Linear N,K=", N, ",", K, " used_nthr=", used_nthr, "  do_splitK=", do_splitK);

        ov::parallel_nt_static(0, [&](const size_t ithr, const size_t nthr) {
            auto& work = works[ithr];
            if (work) {
                work.setup(p_weight, stride);
            }
        });
        DEBUG_LOG("   setup is done. weight @ ", static_cast<void*>(p_weight));
    }

    void run(uint8_t* pA, int strideA, int M, ov::bfloat16* dstC, int strideC) {
        static ReduceAdd2bh jit_reduce2bh_1(false);
        static ReduceAdd2bh jit_reduce2bh_2(true);

        ov::parallel_nt_static(0, [&](const size_t ithr, const size_t nthr) {
            auto& work = works[ithr];
            auto& workC = getC(ithr);
            if (work) {
                work.run(M, pA, strideA, workC);

                if (do_splitK) {
                    auto sync_id = work.sync_flag->fetch_add(1);
                    // (0,1) (2,3)
                    if (sync_id & 1) {
                        auto peer_ithr = (ithr & 1) ? (ithr - 1) : (ithr + 1);
                        auto& peerC = getC(peer_ithr);
                        // the other one has finished, we can do the reduce sum
                        auto* src0 = workC.ptr<float>();
                        auto* src1 = peerC.ptr<float>();
                        auto* dst = dstC + work.n0;
                        auto strideS = workC.stride(0);
                        auto strideD = strideC / sizeof(*dst);
                        for (int m = 0; m < M; m++, src0 += strideS, src1 += strideS, dst += strideD) {
                            // the prefetch distance is increased to ensure by the time store happens
                            // prefetch has done and no HW prefetcher is triggered
                            auto* prefetch_dst = (m + 2 < M) ? (dst + 2 * strideD) : (dst);
                            jit_reduce2bh_2(src0, src1, dst, prefetch_dst, work.BN);
                        }
                    }
                } else {
                    auto* src = workC.ptr<float>();
                    auto* dst = dstC + work.n0;
                    auto strideS = workC.stride(0);
                    auto strideD = strideC / sizeof(*dst);
                    for (int m = 0; m < M; m++, src += strideS, dst += strideD) {
                        // the prefetch distance is increased to ensure by the time store happens
                        // prefetch has done and no HW prefetcher is triggered
                        auto* prefetch_dst = (m + 2 < M) ? (dst + 2 * strideD) : (dst);
                        jit_reduce2bh_1(src, dst, prefetch_dst, work.BN);
                    }
                }
            }
        });
    }

    // gate & up are interleaved: 16 gates + 16 up
    void runGateUp(uint8_t* pA, int strideA, int M, ov::bfloat16* dstC, int strideC, const LLMMLPNode::Config& config) {
        static GateUpCombine jit_gateup_silu(dnnl_eltwise_swish);
        static GateUpCombine jit_gateup_gelu(dnnl_eltwise_gelu_tanh);

        GateUpCombine* jit_gateup;
        if (config.is_act_gelu)
            jit_gateup = &jit_gateup_gelu;
        else if (config.is_act_silu)
            jit_gateup = &jit_gateup_silu;
        else
            OPENVINO_THROW("unsupported act in GateUpCombine");

        ov::parallel_nt_static(0, [&](const size_t ithr, const size_t nthr) {
            auto& work = works[ithr];
            auto& workC = getC(ithr);
            if (work.BN > 0) {
                work.run(M, pA, strideA, workC);
                // K reduce is done, results of [M, BN] sub-block is ready in L2.
                // combine Gate & Up
                auto* src = workC.ptr<float>();
                auto strideS = workC.stride(0);
                auto* dst = dstC + (work.n0 / 2);  // important output is only half of the total N
                auto strideD = strideC / sizeof(*dst);
                for (int m = 0; m < M; m++, src += strideS, dst += strideD) {
                    auto* prefetch_dst = (m + 1 < M) ? (dst + strideD) : (dst);
                    (*jit_gateup)(src, dst, prefetch_dst, work.BN);
                }
            }
        });
    }
};

struct QKVProj : public LLMMLP::Executor {
    const LLMQKVProjNode::Config m_config;
    std::vector<Work> works;

    // q k v each have 1/3 or worker-thread
    QKVProj(ov::bfloat16* wq, ov::bfloat16* wk, ov::bfloat16* wv, const LLMQKVProjNode::Config& config) : m_config(config) {
        const int blk_K_size = 256;
        auto N = m_config.hidden_size;
        auto K = m_config.hidden_size;
        // prepare weights, split N among threads
        // in unit of 32
        OPENVINO_ASSERT((N % 32) == 0);
        OPENVINO_ASSERT((K % blk_K_size) == 0);
        auto nthr = parallel_get_max_threads();
        auto num_blk_N = N / 32;
        auto num_blk_K = K / blk_K_size;
        works.resize(nthr);

        int stride = K * sizeof(*wq);

        // every thread should do same amount of work, and some cores can be idle
        auto valid_nthr = nthr / 3;

        int cur_work_id = 0;
        auto create_works = [&](ov::bfloat16* pw, int output_id) {
            // split task on more cores is better on TBB
            auto blkN_per_thread = (num_blk_N) / valid_nthr;
            auto blkN_leftover = num_blk_N - (blkN_per_thread * valid_nthr);
            auto start_blkN = 0;

            for (int ithr = 0; ithr < valid_nthr; ithr++) {
                auto blkN = std::min(num_blk_N - start_blkN, blkN_per_thread);
                if (blkN_leftover > 0) {
                    blkN_leftover--;
                    blkN++;
                }
                if (blkN) {
                    auto& work = works[cur_work_id++];
                    work.blk_K_size = blk_K_size;

                    work.n0 = (start_blkN)*32;
                    work.n1 = (start_blkN + blkN) * 32;
                    work.BN = blkN * 32;
                    work.k0 = 0;
                    work.k1 = blk_K_size * num_blk_K;

                    work.weights.resize(num_blk_K);
                    for (auto& weight : work.weights)
                        weight.resize<ov::bfloat16>({static_cast<size_t>(blkN), static_cast<size_t>(blk_K_size * 32)});

                    work.output_id = output_id;
                    work.p_raw_weights = pw;
                }
                start_blkN += blkN;
            }
        };
        create_works(wq, 0);
        create_works(wk, 1);
        create_works(wv, 2);

        DEBUG_LOG("QKVProj N,K=", N, ",", K, " used_nthr=", cur_work_id);
        ov::parallel_nt_static(0, [&](const size_t ithr, const size_t nthr) {
            auto& work = works[ithr];
            if (work) {
                work.setup(work.p_raw_weights, stride);
            }
        });
        DEBUG_LOG("   setup is done. weight @ ",
                  static_cast<void*>(wq),
                  ",",
                  static_cast<void*>(wk),
                  ",",
                  static_cast<void*>(wv));
    }

    void run(uint8_t* pA,
             int strideA,
             int M,
             ov::bfloat16* dst_q,
             int stride_q,
             ov::bfloat16* dst_k,
             int stride_k,
             ov::bfloat16* dst_v,
             int stride_v) {
        static ReduceAdd2bh jit_2bh(false);
        for (int m = 0; m < M;) {
            int BM = std::min(M - m, 256);

            ov::parallel_nt_static(0, [&](const size_t ithr, const size_t nthr) {
                auto& work = works[ithr];
                auto& C = getC(ithr);
                if (work.BN > 0) {
                    work.run(BM, pA, strideA, C);

                    // compress accumulation result into target
                    auto* src = C.ptr<float>();
                    auto stride_src = C.stride(0);
                    ov::bfloat16* dst = nullptr;
                    int stride_dst = 0;
                    if (work.output_id == 0) {
                        dst = dst_q + work.n0;
                        stride_dst = stride_q / sizeof(*dst);
                    }
                    if (work.output_id == 1) {
                        dst = dst_k + work.n0;
                        stride_dst = stride_k / sizeof(*dst);
                    }
                    if (work.output_id == 2) {
                        dst = dst_v + work.n0;
                        stride_dst = stride_v / sizeof(*dst);
                    }

                    for (int mi = 0; mi < BM; mi++, src += stride_src, dst += stride_dst) {
                        // the prefetch distance is increased to ensure by the time store happens
                        // prefetch has done and no HW prefetcher is triggered
                        auto* prefetch_dst = (mi + 2 < BM) ? (dst + 2 * stride_dst) : (dst);
                        jit_2bh(src, dst, prefetch_dst, work.BN);
                    }
                }
            });
            m += BM;
            pA += BM * strideA;
            dst_q += BM * stride_q / sizeof(ov::bfloat16);
            dst_k += BM * stride_k / sizeof(ov::bfloat16);
            dst_v += BM * stride_v / sizeof(ov::bfloat16);
        }
    }

    void execute(LLMMLP* pnode) override {
        auto input = pnode->getSrcMemoryAtPort(0);
        const auto& ishape = input->getStaticDims();
        uint8_t* pA = input->getDataAs<uint8_t>();
        auto hidden_size = m_config.hidden_size;
        int strideA = hidden_size * 2;
        int M = shape_size(ishape) / ishape[ishape.size() - 1];
        auto* dst0 = pnode->getDstMemoryAtPort(0)->getDataAs<ov::bfloat16>();
        auto* dst1 = pnode->getDstMemoryAtPort(1)->getDataAs<ov::bfloat16>();
        auto* dst2 = pnode->getDstMemoryAtPort(2)->getDataAs<ov::bfloat16>();

        run(pA, strideA, M, dst0, hidden_size * 2, dst1, hidden_size * 2, dst2, hidden_size * 2);
    }
};

struct MLP : LLMMLP::Executor {
    const LLMMLPNode::Config m_config;
    Linear gate_up;
    Linear down;
    int m_N;
    int m_M = 0;

    // MLP is not supposed to run in parallel
    PlainTensor& get_actUp() {
        static PlainTensor actUp;
        return actUp;
    }

    // [M, K] x [N, K] => [M, N] x [K, N] => [M, K]
    // w_gate/w_up : [N, K]
    //     w_down  : [K, N]
    MLP(PlainTensor w_gate, PlainTensor w_up, PlainTensor w_down, const LLMMLPNode::Config& config) : m_config(config) {
        // [N, K] [N, K] interleave (16-16-...) into [2*N, K]
        auto K = m_config.hidden_size;
        auto N = m_config.intermediate_size;
        static PlainTensor w_gate_up;
        w_gate_up.resize<ov::bfloat16>({static_cast<size_t>(2 * N), static_cast<size_t>(K)});
        for (int n = 0; n < N; n += 16) {
            for (int i = 0; i < 16; i++)
                memcpy(w_gate_up.ptr_v(2 * n + i, 0), w_gate.ptr_v(n + i, 0), K * sizeof(ov::bfloat16));
            for (int i = 0; i < 16; i++)
                memcpy(w_gate_up.ptr_v(2 * n + 16 + i, 0), w_up.ptr_v(n + i, 0), K * sizeof(ov::bfloat16));
        }
        gate_up.setup(w_gate_up.ptr<ov::bfloat16>(), w_gate_up.stride_bytes(0), N * 2, K);
        down.setup(w_down.ptr<ov::bfloat16>(), w_down.stride_bytes(0), K, N, true);
        m_N = N;
    }

    void setM(int M) {
        if (m_M < M) {
            get_actUp().resize<ov::bfloat16>({static_cast<size_t>(M), static_cast<size_t>(m_N)});
            m_M = M;
        }
    }

    void run(uint8_t* pA, int strideA, int M, ov::bfloat16* dstC, int strideC) {
        auto& actUp = get_actUp();
        for (int m = 0; m < M;) {
            int BM = std::min(M - m, 512);
            setM(BM);

            gate_up.runGateUp(pA, strideA, BM, actUp.ptr<ov::bfloat16>(), actUp.stride_bytes(0), m_config);
            down.run(reinterpret_cast<uint8_t*>(actUp.ptr<ov::bfloat16>()), actUp.stride_bytes(0), BM, dstC, strideC);

            m += BM;
            pA += BM * strideA;
            dstC += BM * strideC / sizeof(ov::bfloat16);
        }
    }

    void execute(LLMMLP* pnode) override {
        auto input = pnode->getSrcMemoryAtPort(0);
        const auto& ishape = input->getStaticDims();
        uint8_t* pA = input->getDataAs<uint8_t>();
        auto hidden_size = m_config.hidden_size;
        int strideA = hidden_size * sizeof(ov::bfloat16);
        int M = shape_size(ishape) / ishape[ishape.size() - 1];

        auto* dst0 = pnode->getDstMemoryAtPort(0)->getDataAs<ov::bfloat16>();
        auto output = pnode->getDstMemoryAtPort(0);
        int strideC = hidden_size * sizeof(ov::bfloat16);

        run(pA, strideA, M, dst0, strideC);
    }
};
#endif

};  // namespace

LLMMLP::LLMMLP(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr context)
    : Node(op, context, NgraphShapeInferFactory(op, EMPTY_PORT_MASK)),
      m_executor(nullptr) {
    std::string errorMessage;
    if (!isSupportedOperation(op, errorMessage)) {
        OPENVINO_THROW("CPU: " + errorMessage);
    }
    m_is_mlp = false;
    m_is_qkv_proj = false;

    if (const auto node_mlp = std::dynamic_pointer_cast<const LLMMLPNode>(op)) {
        m_is_mlp = true;
        m_config.mlp = node_mlp->get_config();
    } else if (const auto node_qkv = std::dynamic_pointer_cast<const LLMQKVProjNode>(op)) {
        m_is_qkv_proj = true;
        m_config.qkv = node_qkv->get_config();
    } else {
        OPENVINO_THROW("CPU: LLMMLP got unsupported node.");
    }
}

void LLMMLP::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty())
        return;

    auto rtPrecision = ov::element::bf16;
    auto weightPrecision = ov::element::bf16;

    // initialize input ports
    std::vector<PortConfigurator> inPortConfigs;
    inPortConfigs.emplace_back(LayoutType::ncsp, rtPrecision, getInputShapeAtPort(0), false, -1);      // input
    inPortConfigs.emplace_back(LayoutType::ncsp, weightPrecision, getInputShapeAtPort(1), false, -1);  // gate
    inPortConfigs.emplace_back(LayoutType::ncsp, weightPrecision, getInputShapeAtPort(2), false, -1);  // up
    inPortConfigs.emplace_back(LayoutType::ncsp, weightPrecision, getInputShapeAtPort(3), false, -1);  // down

    // initialize output port
    std::vector<PortConfigurator> outPortConfigs;

    if (m_is_qkv_proj) {
        outPortConfigs.emplace_back(LayoutType::ncsp, rtPrecision, getOutputShapeAtPort(0), false, -1);
        outPortConfigs.emplace_back(LayoutType::ncsp, rtPrecision, getOutputShapeAtPort(0), false, -1);
        outPortConfigs.emplace_back(LayoutType::ncsp, rtPrecision, getOutputShapeAtPort(0), false, -1);
    } else if (m_is_mlp) {
        outPortConfigs.emplace_back(LayoutType::ncsp, rtPrecision, getOutputShapeAtPort(0), false, -1);
    }

    addSupportedPrimDesc(inPortConfigs, outPortConfigs, impl_desc_type::ref_any);
}

void LLMMLP::prepareParams() {
#if defined(OPENVINO_ARCH_X86_64)
    if (!m_executor) {
        if (m_is_qkv_proj) {
            auto exec = std::make_shared<QKVProj>(
                        getSrcMemoryAtPort(1)->getDataAs<ov::bfloat16>(),
                        getSrcMemoryAtPort(2)->getDataAs<ov::bfloat16>(),
                        getSrcMemoryAtPort(3)->getDataAs<ov::bfloat16>(),
                        m_config.qkv);
            m_executor = exec;
        } else if (m_is_mlp) {
            auto exec = std::make_shared<MLP>(getSrcMemoryAtPort(1),
                                              getSrcMemoryAtPort(2),
                                              getSrcMemoryAtPort(3),
                                              m_config.mlp);
            m_executor = exec;
        }
    }
#endif
}

void LLMMLP::execute(dnnl::stream strm) {
    MAYBE_UNUSED(strm);
#if defined(OPENVINO_ARCH_X86_64)
    m_executor->execute(this);
#endif
}

bool LLMMLP::isSupportedOperation(const std::shared_ptr<const ov::Node>& op, std::string& errorMessage) noexcept {
    try {
        const auto node_mlp = std::dynamic_pointer_cast<const LLMMLPNode>(op);
        const auto node_qkv = std::dynamic_pointer_cast<const LLMQKVProjNode>(op);
        if (!node_mlp && !node_qkv) {
            errorMessage = "Only LLMMLPNode or LLMQKVProjNode operation is supported";
            return false;
        }
    } catch (...) {
        return false;
    }
    return true;
}

}  // namespace node
}  // namespace intel_cpu
}  // namespace ov
