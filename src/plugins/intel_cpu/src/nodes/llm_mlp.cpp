// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "llm_mlp.h"

#include <string>
#include <utility>
#include <vector>

#include "common/bfloat16.hpp"
#include "common/cpu_memcpy.h"
#include "cpu/x64/cpu_isa_traits.hpp"
#include "cpu/x64/jit_generator.hpp"
#include "openvino/core/parallel.hpp"
#include "shape_inference/shape_inference_internal_dyn.hpp"
#include "utils/plain_tensor.hpp"

#if defined(OPENVINO_ARCH_X86_64)
#    include "kernels/x64/mlp_kernel.hpp"
#    include "kernels/x64/mlp_utils.hpp"
#endif

namespace ov::intel_cpu::node {

#if defined(OPENVINO_ARCH_X86_64)

template <typename T>
class LinearKsplit2 {
public:
    std::vector<Work> works;

    int used_nthr = 0;

    WeightBuffer wbuffer;

    LinearKsplit2() = default;

    // weight [N, K]
    // Gate & Up are interleaved in N dimension: 16-gate / 16-up
    // and post-ops will compute  silu(gate)*up in unit of 16 elements
    // and store out as bfloat16.
    void setup(void* p_weight, int stride, int N, int K, const LLMMLPNode::Config& config) {
        bool is_quantized = config.down_quantized;

        auto reg_blk_K_size = is_quantized ? REG_BLK_K_SIZE_I8 : REG_BLK_K_SIZE;
        auto cache_blk_k_size = CACHE_BLK_K_SIZE;
        auto weight_element_size = is_quantized ? sizeof(int8_t) : sizeof(ov::float16);

        OPENVINO_ASSERT((N % REG_BLK_N_SIZE) == 0);
        OPENVINO_ASSERT((K % reg_blk_K_size) == 0);
        m_threads_num = parallel_get_max_threads();
        auto num_blk_N = N / REG_BLK_N_SIZE;
        works.resize(m_threads_num);

        auto K_splits = 2;
        // split task on more cores is better on TBB
        auto valid_nthr = m_threads_num / 2;
        auto blkN_per_thread = (num_blk_N) / valid_nthr;
        auto blkN_leftover = num_blk_N - (blkN_per_thread * valid_nthr);
        auto start_blkN = 0;
        used_nthr = 0;

        for (int ithr = 0; ithr < m_threads_num; ithr += K_splits) {
            auto blkN = std::min(num_blk_N - start_blkN, blkN_per_thread);
            if (blkN_leftover > 0) {
                blkN_leftover--;
                blkN++;
            }
            if (blkN) {
                auto shared_atomic = std::make_shared<std::atomic_int>(0);

                // split K dimension in unit of 32 evenly among 2 worker-threads
                auto start_blkK = 0;
                auto num_blk_K = K / reg_blk_K_size;
                auto blkK_per_thread = (num_blk_K + 1) / 2;
                for (int ik = 0; ik < K_splits; ik++) {
                    auto blk_K = std::min(num_blk_K - start_blkK, blkK_per_thread);

                    auto& work = works[ithr + ik];

                    work.sync_flag = shared_atomic;
                    work.blk_K_size = cache_blk_k_size;

                    work.n0 = (start_blkN)*REG_BLK_N_SIZE;
                    work.n1 = (start_blkN + blkN) * REG_BLK_N_SIZE;
                    work.BN = blkN * REG_BLK_N_SIZE;
                    work.k0 = start_blkK * reg_blk_K_size;
                    work.k1 = (start_blkK + blk_K) * reg_blk_K_size;
                    work.quant_i8 = is_quantized;
                    work.is_f16 = std::is_same<T, ov::float16>::value;

                    start_blkK += blk_K;
                    used_nthr++;
                }
            }

            start_blkN += blkN;
        }

        DEBUG_LOG("Linear N,K=", N, ",", K, " used_nthr=", used_nthr);

        wbuffer.alloc(works, weight_element_size);

        ov::parallel_nt_static(m_threads_num, [&](const size_t ithr, const size_t nthr) {
            auto& work = works[ithr];
            if (work) {
                if (is_quantized) {
                    work.setup(wbuffer.get<int8_t>(ithr), reinterpret_cast<int8_t*>(p_weight), stride, true);
                } else {
                    work.setup(wbuffer.get<T>(ithr), reinterpret_cast<ov::float16*>(p_weight), stride);
                }
            }
        });
        DEBUG_LOG("   setup is done. weight @ ", static_cast<void*>(p_weight));
    }

    void run(uint8_t* pA,
             int strideA,
             int M,
             T* dstC,
             int strideC,
             const LLMMLPNode::Config& config,
             MatrixDynQuantPerRow& src_dq,
             float* w_scale) {
        static ReduceAdd2bh jit_reduce2cvt(true, std::is_same<T, ov::float16>::value);

        ov::parallel_nt_static(m_threads_num, [&](const size_t ithr, const size_t nthr) {
            auto& work = works[ithr];
            auto& workC = work.m_C;
            if (work) {
                work.run(M, pA, strideA);

                if (config.down_quantized) {
                    // de-quantize i32 results in-place into f32
                    auto* ptr_c = work.m_C.template ptr<float>();
                    auto* ptr_wsum = work.w_sum_per_oc.template ptr<float>();
                    auto stride_c = work.m_C.stride(0);
                    ov::Extensions::Cpu::XARCH::llm_mlp_dequantize_i32_f32(M,
                                                                           work.BN,
                                                                           reinterpret_cast<int32_t*>(ptr_c),
                                                                           stride_c,
                                                                           ptr_c,
                                                                           stride_c,
                                                                           src_dq.scale,
                                                                           src_dq.zp,
                                                                           ptr_wsum,
                                                                           w_scale + work.n0,
                                                                           src_dq.asym);
                }

                auto sync_id = work.sync_flag->fetch_add(1);
                // (0,1) (2,3)
                if (sync_id & 1) {
                    auto peer_ithr = (ithr & 1) ? (ithr - 1) : (ithr + 1);
                    auto* p_peerC = works[peer_ithr].m_C.template ptr<float>();
                    // the other one has finished, we can do the reduce sum
                    auto* p_curC = workC.template ptr<float>();
                    jit_reduce2cvt
                        .call(p_curC, p_peerC, workC.stride(0), dstC + work.n0, strideC / sizeof(*dstC), M, work.BN);
                }
            }
        });
    }

private:
    int m_threads_num = 0;
};

template <typename T>
class LinearGateUp {
public:
    std::vector<Work> works;

    int used_nthr = 0;

    LinearGateUp() = default;

    WeightBuffer wbuffer;

    GateUpCombine* jit_gateup;

    // weight [N, K]
    // Gate & Up are interleaved in N dimension: 16-gate / 16-up
    // and post-ops will compute  silu(gate)*up in unit of 16 elements
    // and store out as bfloat16.
    void setup(void* p_weight_gate, void* p_weight_up, int stride, int N, int K, const LLMMLPNode::Config& config) {
        static GateUpCombine jit_gateup_silu(dnnl_eltwise_swish, std::is_same<T, ov::float16>::value);
        static GateUpCombine jit_gateup_gelu(dnnl_eltwise_gelu_tanh, std::is_same<T, ov::float16>::value);

        if (config.act == LLMMLPNode::ACT_FN::GELU) {
            jit_gateup = &jit_gateup_gelu;
        } else if (config.act == LLMMLPNode::ACT_FN::SILU) {
            jit_gateup = &jit_gateup_silu;
        } else {
            OPENVINO_THROW("unsupported act in GateUpCombine");
        }

        bool quantized_int8 = config.gate_up_quantized;

        auto reg_blk_K_size = quantized_int8 ? REG_BLK_K_SIZE_I8 : REG_BLK_K_SIZE;
        auto cache_blk_k_size = CACHE_BLK_K_SIZE;
        auto weight_element_size = quantized_int8 ? sizeof(int8_t) : sizeof(ov::float16);

        // prepare weights, split N among threads
        // in unit of 32
        OPENVINO_ASSERT((N % REG_BLK_N_SIZE) == 0);
        OPENVINO_ASSERT((K % reg_blk_K_size) == 0);
        m_threads_num = parallel_get_max_threads();
        auto num_blk_N = N / REG_BLK_N_SIZE;
        works.resize(m_threads_num);

        // split task on more cores is better on TBB
        auto valid_nthr = m_threads_num;
        auto blkN_per_thread = (num_blk_N) / valid_nthr;
        auto blkN_leftover = num_blk_N - (blkN_per_thread * valid_nthr);
        auto start_blkN = 0;
        used_nthr = 0;

        for (int ithr = 0; ithr < m_threads_num; ithr++) {
            auto blkN = std::min(num_blk_N - start_blkN, blkN_per_thread);
            if (blkN_leftover > 0) {
                blkN_leftover--;
                blkN++;
            }
            if (blkN) {
                auto& work = works[ithr];
                work.sync_flag = std::make_shared<std::atomic_int>(0);
                work.blk_K_size = cache_blk_k_size;

                work.n0 = (start_blkN)*REG_BLK_N_SIZE;
                work.n1 = (start_blkN + blkN) * REG_BLK_N_SIZE;
                work.BN = blkN * REG_BLK_N_SIZE;
                work.k0 = 0;
                work.k1 = K;
                work.quant_i8 = quantized_int8;
                work.is_f16 = std::is_same<T, ov::float16>::value;
                used_nthr++;
            }

            start_blkN += blkN;
        }
        wbuffer.alloc(works, weight_element_size);

        DEBUG_LOG("Linear N,K=", N, ",", K, " used_nthr=", used_nthr);
        ov::parallel_nt_static(m_threads_num, [&](const size_t ithr, const size_t nthr) {
            auto& work = works[ithr];
            if (work) {
                if (quantized_int8) {
                    work.setup(wbuffer.get<int8_t>(ithr),
                               reinterpret_cast<int8_t*>(p_weight_gate),
                               reinterpret_cast<int8_t*>(p_weight_up),
                               stride,
                               true);
                } else {
                    work.setup(wbuffer.get<T>(ithr),
                               reinterpret_cast<ov::float16*>(p_weight_gate),
                               reinterpret_cast<ov::float16*>(p_weight_up),
                               stride);
                }
            }
        });
        DEBUG_LOG("   setup is done. weight @ ", static_cast<void*>(p_weight_gate));
    }

    // gate & up are interleaved: 16 gates + 16 up
    void runGateUp(uint8_t* pA,
                   int strideA_in_bytes,
                   int M,
                   T* dstC,
                   int strideC,
                   const LLMMLPNode::Config& config,
                   MatrixDynQuantPerRow& src_dq,
                   float* w_scale) {
        ov::parallel_nt_static(m_threads_num, [&](const size_t ithr, const size_t nthr) {
            auto& work = works[ithr];
            if (work) {
                work.run(M, pA, strideA_in_bytes);

                // K reduce is done, results of [M, BN] sub-block is ready in L2.
                // combine Gate & Up
                float* ptr_c;
                size_t stride_c;
                if (config.gate_up_quantized) {
                    // dequantize m_C in-place
                    ptr_c = work.m_C.template ptr<float>();
                    stride_c = work.m_C.stride(0);
                    auto* p_wsum = work.w_sum_per_oc.template ptr<float>();
                    ov::Extensions::Cpu::XARCH::llm_mlp_dequantize_i32_f32(M,
                                                                           work.BN,
                                                                           reinterpret_cast<int32_t*>(ptr_c),
                                                                           stride_c,
                                                                           ptr_c,
                                                                           stride_c,
                                                                           src_dq.scale,
                                                                           src_dq.zp,
                                                                           p_wsum,
                                                                           w_scale + work.n0,
                                                                           src_dq.asym);
                } else {
                    ptr_c = work.m_C.template ptr<float>();
                    stride_c = work.m_C.stride(0);
                }
                jit_gateup->call(ptr_c, stride_c, dstC + (work.n0 / 2), strideC / sizeof(*dstC), M, work.BN);
            }
        });
    }

private:
    int m_threads_num = 0;
};

template <typename T>
struct LLMMLP::Executor : public LLMMLP::ExecutorBase {
    LLMMLP* m_pnode;
    const LLMMLPNode::Config m_config;
    DnnlScratchPadPtr m_scrachPad;
    MemoryPtr m_scratchMem;
    uint8_t* m_scratch_base = nullptr;

    LinearGateUp<T> gate_up;
    LinearKsplit2<T> down;
    int m_N;
    int m_M = 0;

    // MLP is not supposed to run in parallel
    PlainTensor m_actUp;

    // quantized input: in scratch buffer
    MatrixDynQuantPerRow m_quant_act;
    MatrixDynQuantPerRow m_quant_up_act;

    PlainTensor m_w_scale_gateup;

    bool m_rt_prec_f16;

    // [M, K] x [N, K] => [M, N] x [K, N] => [M, K]
    // w_gate/w_up : [N, K]
    //     w_down  : [K, N]
    Executor(LLMMLP* pnode, const LLMMLPNode::Config& config, DnnlScratchPadPtr scrachPad)
        : m_pnode(pnode),
          m_config(config),
          m_scrachPad(std::move(scrachPad)),
          m_rt_prec_f16(std::is_same<T, ov::float16>::value) {
        PlainTensor w_gate(pnode->getSrcMemoryAtPort(1));
        PlainTensor w_up(pnode->getSrcMemoryAtPort(2));
        PlainTensor w_down(pnode->getSrcMemoryAtPort(3));

        // [N, K] [N, K] interleave (16-16-...) into [2*N, K]
        auto K = w_gate.size(1);
        auto N = w_gate.size(0);
        OPENVINO_ASSERT(w_gate.stride_bytes(0) == w_up.stride_bytes(0));
        if (m_config.gate_up_combined) {
            N = w_gate.size(0) / 2;
            gate_up.setup(w_gate.ptr_v(), w_up.ptr_v(N, 0), w_up.stride_bytes(0), N * 2, K, config);
        } else {
            gate_up.setup(w_gate.ptr_v(), w_up.ptr_v(), w_up.stride_bytes(0), N * 2, K, config);
        }
        down.setup(w_down.ptr_v(), w_down.stride_bytes(0), K, N, config);

        if (m_config.gate_up_quantized) {
            m_w_scale_gateup.resize<float>({N * 2});
            auto* w_scale_gate = pnode->getSrcMemoryAtPort(4)->getDataAs<float>();
            auto* w_scale_up = pnode->getSrcMemoryAtPort(5)->getDataAs<float>();
            auto* dst = m_w_scale_gateup.ptr<float>();
            if (m_config.gate_up_combined) {
                w_scale_up = w_scale_gate + N;
            }
            for (size_t i = 0; i < N; i += 16) {
                memcpy(dst, w_scale_gate + i, 16 * sizeof(float));
                dst += 16;
                memcpy(dst, w_scale_up + i, 16 * sizeof(float));
                dst += 16;
            }
        }

        m_N = N;
    }

    void setM(int M) {
        uint8_t* cur_scratch_base = nullptr;
        if (m_scratchMem) {
            cur_scratch_base = m_scratchMem->getDataAs<uint8_t>();
        }
        // new M larger than previous or the scratch pointer is changed after the following allocation
        if (m_M < M || cur_scratch_base != m_scratch_base) {
            ScratchBuffAllocator allocator;

            allocator.register_allocation(M * m_N * sizeof(T), [&](void* ptr) {
                m_actUp.resize<T>({static_cast<size_t>(M), static_cast<size_t>(m_N)}, reinterpret_cast<T*>(ptr));
            });

            m_threads_num = parallel_get_max_threads();
            for (size_t ithr = 0lu; ithr < m_threads_num; ithr++) {
                auto C1_size = gate_up.works[ithr].set_C(M, reinterpret_cast<float*>(cur_scratch_base));
                auto C2_size = down.works[ithr].set_C(M, reinterpret_cast<float*>(cur_scratch_base));
                auto max_C_size = std::max(C1_size, C2_size);
                allocator.register_allocation(max_C_size, [this, ithr, M](void* ptr) {
                    // these two op runs at different time step, so can share same scratch buffer
                    gate_up.works[ithr].set_C(M, reinterpret_cast<float*>(ptr));
                    down.works[ithr].set_C(M, reinterpret_cast<float*>(ptr));
                });
            }

            if (m_config.gate_up_quantized) {
                m_quant_act.M = M;
                m_quant_act.K = m_config.hidden_size;
                allocator.register_allocation(m_quant_act.size(), [&](void* ptr) {
                    m_quant_act.setup(ptr);
                });
            }

            if (m_config.down_quantized) {
                m_quant_up_act.M = M;
                m_quant_up_act.K = m_config.up_size;
                allocator.register_allocation(m_quant_up_act.size(), [&](void* ptr) {
                    m_quant_up_act.setup(ptr);
                });
            }

            auto newMemDesc = std::make_shared<CpuBlockedMemoryDesc>(ov::element::u8, Shape{allocator.size()});
            m_scratchMem = m_scrachPad->createScratchPadMem(newMemDesc);
            m_scratch_base = m_scratchMem->getDataAs<uint8_t>();

            allocator.finalize(m_scratch_base);
            m_M = M;
        }
    }

    void execute() override {
        auto input = m_pnode->getSrcMemoryAtPort(0);
        const auto& ishape = input->getStaticDims();
        auto* pA = input->getDataAs<uint8_t>();
        const auto& srcStrides = input->getDescWithType<BlockedMemoryDesc>()->getStrides();

        int strideA = srcStrides[srcStrides.size() - 2];
        int strideA_in_bytes = strideA * sizeof(T);
        int M = shape_size(ishape) / ishape[ishape.size() - 1];

        auto output = m_pnode->getDstMemoryAtPort(0);
        auto* dstC = output->getDataAs<T>();
        const auto& dstStrides = output->getDescWithType<BlockedMemoryDesc>()->getStrides();
        int strideC = dstStrides[dstStrides.size() - 2] * sizeof(T);

        float* p_w_scale_down = nullptr;
        if (m_config.down_quantized) {
            p_w_scale_down = m_pnode->getSrcMemoryAtPort(6)->getDataAs<float>();
        }

        for (int m = 0; m < M;) {
            int BM = std::min(M - m, CACHE_BLK_M_SIZE);
            setM(BM);

            uint8_t* psrc = pA;
            auto stride_src_in_bytes = strideA_in_bytes;
            auto strideA_in_bytes = strideA * sizeof(T);
            if (m_config.gate_up_quantized) {
                m_quant_act.quantize(BM, reinterpret_cast<T*>(pA), strideA);
                psrc = reinterpret_cast<uint8_t*>(m_quant_act.data);
                stride_src_in_bytes = m_quant_act.K;
            }

            // dequantize is fused into gate_up
            gate_up.runGateUp(psrc,
                              stride_src_in_bytes,
                              BM,
                              m_actUp.ptr<T>(),
                              m_actUp.stride_bytes(0),
                              m_config,
                              m_quant_act,
                              m_w_scale_gateup.ptr<float>());

            auto* p_up_act = reinterpret_cast<uint8_t*>(m_actUp.ptr<T>());
            size_t stride_up_act = m_actUp.stride_bytes(0);
            if (m_config.down_quantized) {
                m_quant_up_act.quantize(BM, m_actUp.ptr<T>(), m_actUp.stride(0));
                p_up_act = reinterpret_cast<uint8_t*>(m_quant_up_act.data);
                stride_up_act = m_quant_up_act.stride();
            }

            down.run(p_up_act, stride_up_act, BM, dstC, strideC, m_config, m_quant_up_act, p_w_scale_down);

            m += BM;
            pA += BM * strideA_in_bytes;
            dstC += BM * strideC / sizeof(T);
        }
    }

private:
    size_t m_threads_num = 0lu;
};
#else
template <typename T>
struct LLMMLP::Executor : public LLMMLP::ExecutorBase {
    Executor(LLMMLP*, const LLMMLPNode::Config&, const DnnlScratchPadPtr&) {}
    void execute() {}
};
#endif

LLMMLP::LLMMLP(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr& context)
    : Node(op, context, NgraphShapeInferFactory(op)) {
    std::string errorMessage;
    const auto& config = context->getConfig();
    if (!isSupportedOperation(op, errorMessage, config.fcDynamicQuantizationGroupSize)) {
        OPENVINO_THROW_NOT_IMPLEMENTED(errorMessage);
    }
    const auto node_mlp = ov::as_type_ptr<const LLMMLPNode>(op);
    m_mlp_config = node_mlp->get_config();
}

void LLMMLP::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty()) {
        return;
    }

    std::vector<PortConfigurator> inPortConfigs;
    std::vector<PortConfigurator> outPortConfigs;

    auto rtPrecision = getOriginalInputPrecisionAtPort(0);

    if (rtPrecision == ov::element::f32) {
        // fallback to supported precision if possible
        if (dnnl::impl::cpu::x64::mayiuse(dnnl::impl::cpu::x64::avx512_core_amx_fp16)) {
            rtPrecision = ov::element::f16;
        } else if (dnnl::impl::cpu::x64::mayiuse(dnnl::impl::cpu::x64::avx512_core_amx)) {
            rtPrecision = ov::element::bf16;
        }
    }

    OPENVINO_ASSERT(rtPrecision == ov::element::bf16 || rtPrecision == ov::element::f16,
                    "Unexpected rtPrecision:",
                    rtPrecision);

    if (m_mlp_config.gate_up_quantized) {
        auto weightPrecision = ov::element::i8;

        // initialize input ports
        inPortConfigs.emplace_back(LayoutType::ncsp, rtPrecision, getInputShapeAtPort(0), false, -1);      // input
        inPortConfigs.emplace_back(LayoutType::ncsp, weightPrecision, getInputShapeAtPort(1), false, -1);  // gate
        inPortConfigs.emplace_back(LayoutType::ncsp, weightPrecision, getInputShapeAtPort(2), false, -1);  // up
        inPortConfigs.emplace_back(LayoutType::ncsp,
                                   m_mlp_config.down_quantized ? ov::element::i8 : ov::element::f16,
                                   getInputShapeAtPort(3),
                                   false,
                                   -1);  // down
        inPortConfigs.emplace_back(LayoutType::ncsp,
                                   ov::element::f32,
                                   getInputShapeAtPort(4),
                                   false,
                                   -1);  // gate_weight scales per OC
        inPortConfigs.emplace_back(LayoutType::ncsp,
                                   ov::element::f32,
                                   getInputShapeAtPort(5),
                                   false,
                                   -1);  // up_weight scales per OC
        if (m_mlp_config.down_quantized) {
            inPortConfigs.emplace_back(LayoutType::ncsp,
                                       ov::element::f32,
                                       getInputShapeAtPort(6),
                                       false,
                                       -1);  // down_weight scales per OC
        }

        // initialize output port
        outPortConfigs.emplace_back(LayoutType::ncsp, rtPrecision, getOutputShapeAtPort(0), false, -1);
    } else {
        auto weightPrecision = ov::element::f16;

        // initialize input ports
        inPortConfigs.emplace_back(LayoutType::ncsp, rtPrecision, getInputShapeAtPort(0), false, -1);      // input
        inPortConfigs.emplace_back(LayoutType::ncsp, weightPrecision, getInputShapeAtPort(1), false, -1);  // gate
        inPortConfigs.emplace_back(LayoutType::ncsp, weightPrecision, getInputShapeAtPort(2), false, -1);  // up
        inPortConfigs.emplace_back(LayoutType::ncsp, weightPrecision, getInputShapeAtPort(3), false, -1);  // down

        // initialize output port
        outPortConfigs.emplace_back(LayoutType::ncsp, rtPrecision, getOutputShapeAtPort(0), false, -1);
    }
    addSupportedPrimDesc(inPortConfigs, outPortConfigs, impl_desc_type::ref_any);
}

void LLMMLP::createPrimitive() {
    auto rtPrecision = getInputPrecisions()[0];
#ifdef OPENVINO_ARCH_X86_64
    if (rtPrecision == ov::element::bf16) {
        m_executor = std::make_shared<Executor<ov::bfloat16>>(this, m_mlp_config, context->getScratchPad());
    } else if (rtPrecision == ov::element::f16) {
        m_executor = std::make_shared<Executor<ov::float16>>(this, m_mlp_config, context->getScratchPad());
    }
#endif
    if (!m_executor) {
        THROW_CPU_NODE_ERR("Executor creation fails with precision " + rtPrecision.to_string());
    }
}

void LLMMLP::execute(const dnnl::stream& strm) {
    MAYBE_UNUSED(strm);
    m_executor->execute();
}

bool LLMMLP::isSupportedOperation(const std::shared_ptr<const ov::Node>& op,
                                  std::string& errorMessage,
                                  uint64_t fcDynamicQuantizationGroupSize) noexcept {
#if defined(OPENVINO_ARCH_X86_64)
    try {
        const auto node_mlp = ov::as_type_ptr<const LLMMLPNode>(op);
        if (node_mlp) {
            auto down_proj_w_pshape = op->input_value(1).get_partial_shape();
            if (!down_proj_w_pshape.is_static()) {
                // return true to skip Fusion
                errorMessage = "LLMMLPNode weight shape is not static";
                return false;
            }
            auto down_size = down_proj_w_pshape[0].get_length();
            auto up_size = down_proj_w_pshape[1].get_length();

            auto& config = node_mlp->get_config();
            if (config.gate_up_quantized &&
                (fcDynamicQuantizationGroupSize < static_cast<uint64_t>(config.hidden_size))) {
                errorMessage = "LLMMLPNode gate-up-proj only support per-token dynamic quantization";
                return false;
            }

            if (config.down_quantized && (fcDynamicQuantizationGroupSize < static_cast<uint64_t>(config.up_size))) {
                errorMessage = "LLMMLPNode down_proj only support per-token dynamic quantization";
                return false;
            }

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

}  // namespace ov::intel_cpu::node
