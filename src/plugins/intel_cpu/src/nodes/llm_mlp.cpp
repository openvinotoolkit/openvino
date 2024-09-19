// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "llm_mlp.h"

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
#endif

namespace ov {
namespace intel_cpu {
namespace node {

#if defined(OPENVINO_ARCH_X86_64)

class LinearKsplit2 {
public:
    std::vector<Work> works;

    int used_nthr = 0;

    WeightBuffer wbuffer;

    LinearKsplit2() {}

    ReduceAdd2bh * p_jit_reduce2bh;
    // weight [N, K]
    // Gate & Up are interleaved in N dimension: 16-gate / 16-up
    // and post-ops will compute  silu(gate)*up in unit of 16 elements
    // and store out as bfloat16.
    template <typename T>
    void setup(T* p_weight, int stride, int N, int K) {
        static ReduceAdd2bh jit_reduce2bh_2(true);

        OPENVINO_ASSERT((N % REG_BLK_N_SIZE) == 0);
        OPENVINO_ASSERT((K % REG_BLK_K_SIZE) == 0);
        auto nthr = parallel_get_max_threads();
        auto num_blk_N = N / REG_BLK_N_SIZE;
        works.resize(nthr);

        p_jit_reduce2bh = &jit_reduce2bh_2;

        auto K_splits = 2;
        // split task on more cores is better on TBB
        auto valid_nthr = nthr / 2;
        auto blkN_per_thread = (num_blk_N) / valid_nthr;
        auto blkN_leftover = num_blk_N - (blkN_per_thread * valid_nthr);
        auto start_blkN = 0;
        used_nthr = 0;

        for (int ithr = 0; ithr < nthr; ithr += K_splits) {
            auto blkN = std::min(num_blk_N - start_blkN, blkN_per_thread);
            if (blkN_leftover > 0) {
                blkN_leftover--;
                blkN++;
            }
            if (blkN) {
                auto shared_atomic = std::make_shared<std::atomic_int>(0);

                // split K dimension in unit of 32 evenly among 2 worker-threads
                auto start_blkK = 0;
                auto num_blk_K = K / REG_BLK_K_SIZE;
                auto blkK_per_thread = (num_blk_K + 1) / 2;
                for (int ik = 0; ik < K_splits; ik++) {
                    auto blk_K = std::min(num_blk_K - start_blkK, blkK_per_thread);

                    auto& work = works[ithr + ik];

                    work.sync_flag = shared_atomic;
                    work.blk_K_size = CACHE_BLK_K_SIZE;

                    work.n0 = (start_blkN) * REG_BLK_N_SIZE;
                    work.n1 = (start_blkN + blkN) * REG_BLK_N_SIZE;
                    work.BN = blkN * REG_BLK_N_SIZE;
                    work.k0 = start_blkK * REG_BLK_K_SIZE;
                    work.k1 = (start_blkK + blk_K) * REG_BLK_K_SIZE;

                    start_blkK += blk_K;
                    used_nthr++;
                }
            }

            start_blkN += blkN;
        }

        DEBUG_LOG("Linear N,K=", N, ",", K, " used_nthr=", used_nthr);

        wbuffer.alloc(works);

        ov::parallel_nt_static(0, [&](const size_t ithr, const size_t nthr) {
            auto& work = works[ithr];
            if (work) {
                work.setup(wbuffer.get(ithr), p_weight, stride);
            }
        });
        DEBUG_LOG("   setup is done. weight @ ", static_cast<void*>(p_weight));
    }

    void run(uint8_t* pA, int strideA, int M, ov::bfloat16* dstC, int strideC) {
        ov::parallel_nt_static(0, [&](const size_t ithr, const size_t nthr) {
            auto& work = works[ithr];
            auto& workC = work.m_C;
            if (work) {
                work.run(M, pA, strideA);

                auto sync_id = work.sync_flag->fetch_add(1);
                // (0,1) (2,3)
                if (sync_id & 1) {
                    auto peer_ithr = (ithr & 1) ? (ithr - 1) : (ithr + 1);
                    auto& peerC = works[peer_ithr].m_C;
                    // the other one has finished, we can do the reduce sum
                    p_jit_reduce2bh->call(workC.ptr<float>(), peerC.ptr<float>(), workC.stride(0),
                                            dstC + work.n0, strideC / sizeof(*dstC),
                                            M, work.BN);
                }
            }
        });
    }
};

class LinearGateUp {
public:
    std::vector<Work> works;

    int used_nthr = 0;

    LinearGateUp() {}

    WeightBuffer wbuffer;

    GateUpCombine* jit_gateup;

    // weight [N, K]
    // Gate & Up are interleaved in N dimension: 16-gate / 16-up
    // and post-ops will compute  silu(gate)*up in unit of 16 elements
    // and store out as bfloat16.
    template <typename T>
    void setup(T* p_weight_gate, T* p_weight_up, int stride, int N, int K, const LLMMLPNode::Config& config) {
        static GateUpCombine jit_gateup_silu(dnnl_eltwise_swish);
        static GateUpCombine jit_gateup_gelu(dnnl_eltwise_gelu_tanh);

        if (config.act == LLMMLPNode::ACT_FN::GELU)
            jit_gateup = &jit_gateup_gelu;
        else if (config.act == LLMMLPNode::ACT_FN::SILU)
            jit_gateup = &jit_gateup_silu;
        else
            OPENVINO_THROW("unsupported act in GateUpCombine");

        // prepare weights, split N among threads
        // in unit of 32
        OPENVINO_ASSERT((N % REG_BLK_N_SIZE) == 0);
        OPENVINO_ASSERT((K % REG_BLK_K_SIZE) == 0);
        auto nthr = parallel_get_max_threads();
        auto num_blk_N = N / REG_BLK_N_SIZE;
        works.resize(nthr);

        // split task on more cores is better on TBB
        auto valid_nthr = nthr;
        auto blkN_per_thread = (num_blk_N) / valid_nthr;
        auto blkN_leftover = num_blk_N - (blkN_per_thread * valid_nthr);
        auto start_blkN = 0;
        used_nthr = 0;

        for (int ithr = 0; ithr < nthr; ithr ++) {
            auto blkN = std::min(num_blk_N - start_blkN, blkN_per_thread);
            if (blkN_leftover > 0) {
                blkN_leftover--;
                blkN++;
            }
            if (blkN) {
                auto shared_atomic = std::make_shared<std::atomic_int>(0);
                auto& work = works[ithr];
                work.sync_flag = shared_atomic;
                work.blk_K_size = CACHE_BLK_K_SIZE;

                work.n0 = (start_blkN) * REG_BLK_N_SIZE;
                work.n1 = (start_blkN + blkN) * REG_BLK_N_SIZE;
                work.BN = blkN * REG_BLK_N_SIZE;
                work.k0 = 0;
                work.k1 = K;
                used_nthr++;
            }

            start_blkN += blkN;
        }
        wbuffer.alloc(works);

        DEBUG_LOG("Linear N,K=", N, ",", K, " used_nthr=", used_nthr);
        ov::parallel_nt_static(0, [&](const size_t ithr, const size_t nthr) {
            auto& work = works[ithr];
            if (work) {
                work.setup(wbuffer.get(ithr), p_weight_gate, p_weight_up, stride);
            }
        });
        DEBUG_LOG("   setup is done. weight @ ", static_cast<void*>(p_weight_gate));
    }

    // gate & up are interleaved: 16 gates + 16 up
    void runGateUp(uint8_t* pA, int strideA, int M,
                   ov::bfloat16* dstC, int strideC,
                   const LLMMLPNode::Config& config) {
        ov::parallel_nt_static(0, [&](const size_t ithr, const size_t nthr) {
            auto& work = works[ithr];
            if (work) {
                work.run(M, pA, strideA);
                // K reduce is done, results of [M, BN] sub-block is ready in L2.
                // combine Gate & Up
                jit_gateup->call(work.m_C.ptr<float>(), work.m_C.stride(0),
                                 dstC + (work.n0 / 2), strideC / sizeof(*dstC),
                                 M, work.BN);
            }
        });
    }
};

struct LLMMLP::Impl {
    const LLMMLPNode::Config m_config;
    DnnlScratchPadPtr m_scrachPad;
    MemoryPtr m_scratchMem;
    uint8_t* m_scratch_base = nullptr;

    LinearGateUp gate_up;
    LinearKsplit2 down;
    int m_N;
    int m_M = 0;

    // MLP is not supposed to run in parallel
    PlainTensor m_actUp;

    // [M, K] x [N, K] => [M, N] x [K, N] => [M, K]
    // w_gate/w_up : [N, K]
    //     w_down  : [K, N]
    Impl(PlainTensor w_gate, PlainTensor w_up, PlainTensor w_down, const LLMMLPNode::Config& config, DnnlScratchPadPtr scrachPad)
         : m_config(config), m_scrachPad(scrachPad) {
        // [N, K] [N, K] interleave (16-16-...) into [2*N, K]
        auto K = w_gate.size(1);
        auto N = w_gate.size(0);
        OPENVINO_ASSERT(w_gate.stride_bytes(0) == w_up.stride_bytes(0));
        gate_up.setup(w_gate.ptr<ov::float16>(), w_up.ptr<ov::float16>(), w_up.stride_bytes(0), N * 2, K, config);
        down.setup(w_down.ptr<ov::float16>(), w_down.stride_bytes(0), K, N);

        m_N = N;
    }

    void setM(int M) {
        uint8_t* cur_scratch_base = nullptr;
        if (m_scratchMem)
            cur_scratch_base = m_scratchMem->getDataAs<uint8_t>();
        // new M larger than previous or the scratch pointer is changed after the following allocation
        if (m_M < M || cur_scratch_base != m_scratch_base) {
            size_t total_scratch_size = M * m_N * sizeof(ov::bfloat16);
            std::vector<size_t> scratch_offsets;
            auto nthr = parallel_get_max_threads();
            for (int ithr = 0; ithr < nthr; ithr++) {
                scratch_offsets.push_back(total_scratch_size);
                auto C1_size = gate_up.works[ithr].set_C(M, reinterpret_cast<float*>(cur_scratch_base));
                auto C2_size = down.works[ithr].set_C(M, reinterpret_cast<float*>(cur_scratch_base));
                auto max_C_size = std::max(C1_size, C2_size);
                total_scratch_size += max_C_size;
            }

            auto newMemDesc = std::make_shared<CpuBlockedMemoryDesc>(ov::element::u8, Shape{total_scratch_size});
            m_scratchMem = m_scrachPad->createScratchPadMem(newMemDesc);

            m_scratch_base = m_scratchMem->getDataAs<uint8_t>();
            m_actUp.resize<ov::bfloat16>({static_cast<size_t>(M), static_cast<size_t>(m_N)}, reinterpret_cast<ov::bfloat16*>(m_scratch_base));

            for (int ithr = 0; ithr < nthr; ithr++) {
                auto C_base = reinterpret_cast<float*>(m_scratch_base + scratch_offsets[ithr]);
                gate_up.works[ithr].set_C(M, C_base);
                down.works[ithr].set_C(M, C_base);
            }
            m_M = M;
        }
    }

    void execute(LLMMLP* pnode) {
        auto input = pnode->getSrcMemoryAtPort(0);
        const auto& ishape = input->getStaticDims();
        uint8_t* pA = input->getDataAs<uint8_t>();
        const auto& srcStrides = input->getDescWithType<BlockedMemoryDesc>()->getStrides();

        int strideA = srcStrides[srcStrides.size() - 2] * sizeof(ov::bfloat16);
        int M = shape_size(ishape) / ishape[ishape.size() - 1];

        auto output = pnode->getDstMemoryAtPort(0);
        auto* dstC = output->getDataAs<ov::bfloat16>();
        const auto& dstStrides = output->getDescWithType<BlockedMemoryDesc>()->getStrides();
        int strideC = dstStrides[dstStrides.size() - 2] * sizeof(ov::bfloat16);

        for (int m = 0; m < M;) {
            int BM = std::min(M - m, CACHE_BLK_M_SIZE);
            setM(BM);
            gate_up.runGateUp(pA, strideA, BM, m_actUp.ptr<ov::bfloat16>(), m_actUp.stride_bytes(0), m_config);
            down.run(reinterpret_cast<uint8_t*>(m_actUp.ptr<ov::bfloat16>()), m_actUp.stride_bytes(0), BM, dstC, strideC);

            m += BM;
            pA += BM * strideA;
            dstC += BM * strideC / sizeof(ov::bfloat16);
        }
    }
};
#else
struct LLMMLP::Impl {
    Impl(PlainTensor w_gate, PlainTensor w_up, PlainTensor w_down, const LLMMLPNode::Config& config, DnnlScratchPadPtr scrachPad) {}
    void execute(LLMMLP* pnode) {}
};
#endif

LLMMLP::LLMMLP(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr context)
    : Node(op, context, NgraphShapeInferFactory(op, EMPTY_PORT_MASK)) {
    std::string errorMessage;
    if (!isSupportedOperation(op, errorMessage)) {
        OPENVINO_THROW("CPU: " + errorMessage);
    }
    const auto node_mlp = std::dynamic_pointer_cast<const LLMMLPNode>(op);
    m_mlp_config = node_mlp->get_config();
}

void LLMMLP::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty())
        return;

    auto rtPrecision = ov::element::bf16;
    auto weightPrecision = ov::element::f16;

    // initialize input ports
    std::vector<PortConfigurator> inPortConfigs;
    inPortConfigs.emplace_back(LayoutType::ncsp, rtPrecision, getInputShapeAtPort(0), false, -1);      // input
    inPortConfigs.emplace_back(LayoutType::ncsp, weightPrecision, getInputShapeAtPort(1), false, -1);  // gate
    inPortConfigs.emplace_back(LayoutType::ncsp, weightPrecision, getInputShapeAtPort(2), false, -1);  // up
    inPortConfigs.emplace_back(LayoutType::ncsp, weightPrecision, getInputShapeAtPort(3), false, -1);  // down

    // initialize output port
    std::vector<PortConfigurator> outPortConfigs;
    outPortConfigs.emplace_back(LayoutType::ncsp, rtPrecision, getOutputShapeAtPort(0), false, -1);

    addSupportedPrimDesc(inPortConfigs, outPortConfigs, impl_desc_type::ref_any);
}

void LLMMLP::prepareParams() {
    if (!m_pimpl) {
        m_pimpl = std::make_shared<Impl>(getSrcMemoryAtPort(1),
                                         getSrcMemoryAtPort(2),
                                         getSrcMemoryAtPort(3),
                                         m_mlp_config,
                                         context->getScratchPad());
    }
}

void LLMMLP::execute(dnnl::stream strm) {
    MAYBE_UNUSED(strm);
    m_pimpl->execute(this);
}

bool LLMMLP::isSupportedOperation(const std::shared_ptr<const ov::Node>& op, std::string& errorMessage) noexcept {
#if defined(OPENVINO_ARCH_X86_64)
    try {
        const auto node_mlp = std::dynamic_pointer_cast<const LLMMLPNode>(op);
        if (node_mlp) {
            auto down_proj_w_pshape = op->input_value(1).get_partial_shape();
            if (!down_proj_w_pshape.is_static()) {
                // return true to skip Fusion
                errorMessage = "LLMMLPNode weight shape is not static";
                return false;
            }
            auto down_size = down_proj_w_pshape[0].get_length();
            auto up_size = down_proj_w_pshape[1].get_length();
            if (down_size % REG_BLK_K_SIZE) {
                errorMessage = "LLMMLPNode down_proj size is not multiple of register blocking size";
                return false;
            }
            if (up_size % REG_BLK_N_SIZE) {
                errorMessage = "LLMMLPNode up_proj size is not multiple of register blocking size";
                return false;
            }
        } else {
            errorMessage = "Only LLMMLPNode operation is supported";
            return false;
        }
    } catch (...) {
        return false;
    }
    return true;
#else
    return false;
#endif
}

}  // namespace node
}  // namespace intel_cpu
}  // namespace ov
