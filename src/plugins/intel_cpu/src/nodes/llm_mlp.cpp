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

    void run(uint8_t* pA, int strideA, int M, ov::bfloat16* dstC, int strideC, std::vector<PlainTensor>& m_tempC) {
        static ReduceAdd2bh jit_reduce2bh_1(false);
        static ReduceAdd2bh jit_reduce2bh_2(true);

        ov::parallel_nt_static(0, [&](const size_t ithr, const size_t nthr) {
            auto& work = works[ithr];
            auto& workC = m_tempC[ithr];
            if (work) {
                work.run(M, pA, strideA, workC);

                if (do_splitK) {
                    auto sync_id = work.sync_flag->fetch_add(1);
                    // (0,1) (2,3)
                    if (sync_id & 1) {
                        auto peer_ithr = (ithr & 1) ? (ithr - 1) : (ithr + 1);
                        auto& peerC = m_tempC[peer_ithr];
                        // the other one has finished, we can do the reduce sum
                        jit_reduce2bh_2.call(workC.ptr<float>(), peerC.ptr<float>(), workC.stride(0),
                                             dstC + work.n0, strideC / sizeof(*dstC),
                                             M, work.BN);
                    }
                } else {
                    jit_reduce2bh_2.call(workC.ptr<float>(), workC.stride(0),
                                         dstC + work.n0, strideC / sizeof(*dstC),
                                         M, work.BN);
                }
            }
        });
    }

    // gate & up are interleaved: 16 gates + 16 up
    void runGateUp(uint8_t* pA, int strideA, int M,
                   ov::bfloat16* dstC, int strideC,
                   const LLMMLPNode::Config& config,
                   std::vector<PlainTensor>& m_tempC) {
        static GateUpCombine jit_gateup_silu(dnnl_eltwise_swish);
        static GateUpCombine jit_gateup_gelu(dnnl_eltwise_gelu_tanh);

        GateUpCombine* jit_gateup;
        if (config.act == LLMMLPNode::ACT_FN::GELU)
            jit_gateup = &jit_gateup_gelu;
        else if (config.act == LLMMLPNode::ACT_FN::SILU)
            jit_gateup = &jit_gateup_silu;
        else
            OPENVINO_THROW("unsupported act in GateUpCombine");

        ov::parallel_nt_static(0, [&](const size_t ithr, const size_t nthr) {
            auto& work = works[ithr];
            auto& workC = m_tempC[ithr];
            if (work.BN > 0) {
                work.run(M, pA, strideA, workC);
                // K reduce is done, results of [M, BN] sub-block is ready in L2.
                // combine Gate & Up
                jit_gateup->call(workC.ptr<float>(), workC.stride(0),
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

    Linear gate_up;
    Linear down;
    int m_N;
    int m_M = 0;

    // MLP is not supposed to run in parallel
    PlainTensor m_actUp;
    std::vector<PlainTensor> m_tempC;

    // [M, K] x [N, K] => [M, N] x [K, N] => [M, K]
    // w_gate/w_up : [N, K]
    //     w_down  : [K, N]
    Impl(PlainTensor w_gate, PlainTensor w_up, PlainTensor w_down, const LLMMLPNode::Config& config, DnnlScratchPadPtr scrachPad)
         : m_config(config), m_scrachPad(scrachPad) {
        // [N, K] [N, K] interleave (16-16-...) into [2*N, K]
        auto K = w_gate.size(1);
        auto N = w_gate.size(0);
        static PlainTensor w_gate_up;
        w_gate_up.resize<ov::bfloat16>({static_cast<size_t>(2 * N), static_cast<size_t>(K)});
        for (size_t n = 0; n < N; n += 16) {
            for (size_t i = 0; i < 16; i++)
                memcpy(w_gate_up.ptr_v(2 * n + i, 0), w_gate.ptr_v(n + i, 0), K * sizeof(ov::bfloat16));
            for (size_t i = 0; i < 16; i++)
                memcpy(w_gate_up.ptr_v(2 * n + 16 + i, 0), w_up.ptr_v(n + i, 0), K * sizeof(ov::bfloat16));
        }
        gate_up.setup(w_gate_up.ptr<ov::bfloat16>(), w_gate_up.stride_bytes(0), N * 2, K);
        down.setup(w_down.ptr<ov::bfloat16>(), w_down.stride_bytes(0), K, N, true);

        m_tempC.resize(parallel_get_max_threads());
        m_N = N;
    }

    void setM(int M) {
        if (m_M < M) {
            size_t total_scratch_size = M * m_N * sizeof(ov::bfloat16);
            std::vector<size_t> scratch_offsets;
            std::vector<size_t> scratch_C_sizes;
            for (size_t ithr = 0; ithr < m_tempC.size(); ithr++) {
                scratch_offsets.push_back(total_scratch_size);
                auto max_C_size = std::max(gate_up.works[ithr].get_C_size(M), down.works[ithr].get_C_size(M));
                scratch_C_sizes.push_back(max_C_size);
                total_scratch_size += max_C_size * sizeof(float);
            }

            auto newMemDesc = std::make_shared<CpuBlockedMemoryDesc>(ov::element::u8, Shape{total_scratch_size});
            m_scratchMem = m_scrachPad->createScratchPadMem(newMemDesc);

            auto* scratch_base = m_scratchMem->getDataAs<uint8_t>();
            m_actUp.resize<ov::bfloat16>({static_cast<size_t>(M), static_cast<size_t>(m_N)}, reinterpret_cast<ov::bfloat16*>(scratch_base));

            for (size_t ithr = 0; ithr < m_tempC.size(); ithr++) {
                m_tempC[ithr].resize<float>({1, scratch_C_sizes[ithr]}, reinterpret_cast<float*>(scratch_base + scratch_offsets[ithr]));
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
            int BM = std::min(M - m, 512);
            setM(BM);

            gate_up.runGateUp(pA, strideA, BM, m_actUp.ptr<ov::bfloat16>(), m_actUp.stride_bytes(0), m_config, m_tempC);
            down.run(reinterpret_cast<uint8_t*>(m_actUp.ptr<ov::bfloat16>()), m_actUp.stride_bytes(0), BM, dstC, strideC, m_tempC);

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
    auto weightPrecision = ov::element::bf16;

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
            if ((down_size % 256) != 0) {
                errorMessage = "LLMMLPNode down_proj size is not multiple of 256";
                return false;
            }
            if (up_size % 256) {
                errorMessage = "LLMMLPNode up_proj size is not multiple of 256";
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
}

}  // namespace node
}  // namespace intel_cpu
}  // namespace ov
