// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gated_delta_net.h"

#include <common/utils.hpp>
#include <cstddef>
#include <memory>
#include <oneapi/dnnl/dnnl_common.hpp>
#include <string>
#include <vector>

#include "cpu_memory.h"
#include "graph_context.h"
#include "kernels/linear_attn/recurrent_linear_attn.hpp"
#include "memory_desc/cpu_blocked_memory_desc.h"
#include "memory_desc/cpu_memory_desc.h"
#include "node.h"
#include "onednn/iml_type_mapper.h"
#include "openvino/core/except.hpp"
#include "openvino/core/node.hpp"
#include "openvino/core/type.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/op/gated_delta_net.hpp"
#include "shape_inference/shape_inference_cpu.hpp"
#include "utils/general_utils.h"
#include "utils/plain_tensor.hpp"
#if defined(OPENVINO_ARCH_X86_64)
#    include "cpu_parallel.hpp"
#    include "kernels/x64/gdn_jit_kernel.hpp"
#endif

using namespace ov::Extensions::Cpu;
using namespace ov::Extensions::Cpu::XARCH;
using namespace dnnl::impl::cpu::x64;

namespace ov::intel_cpu::node {

#if defined(OPENVINO_ARCH_X86_64)
namespace {
struct GatedDeltaNetKey {
    ov::element::Type precision;
    size_t qk_head_size;
    bool fuse_qk_l2norm;
    float q_l2_norm_eps;
    float k_l2_norm_eps;

    [[nodiscard]] size_t hash() const {
        size_t seed = 0;
        seed = dnnl::impl::hash_combine(seed, precision.hash());
        seed = dnnl::impl::hash_combine(seed, qk_head_size);
        seed = dnnl::impl::hash_combine(seed, fuse_qk_l2norm);
        seed = dnnl::impl::hash_combine(seed, q_l2_norm_eps);
        seed = dnnl::impl::hash_combine(seed, k_l2_norm_eps);
        return seed;
    }

    bool operator==(const GatedDeltaNetKey& rhs) const {
        return precision == rhs.precision && qk_head_size == rhs.qk_head_size && fuse_qk_l2norm == rhs.fuse_qk_l2norm &&
               q_l2_norm_eps == rhs.q_l2_norm_eps && k_l2_norm_eps == rhs.k_l2_norm_eps;
    }
};

void recurrent_linear_attn_jit(const ov::intel_cpu::PlainTensor& query,
                               const ov::intel_cpu::PlainTensor& key,
                               const ov::intel_cpu::PlainTensor& value,
                               const ov::intel_cpu::PlainTensor& recurrent_state,
                               const ov::intel_cpu::PlainTensor& gate,
                               const ov::intel_cpu::PlainTensor& beta,
                               ov::intel_cpu::PlainTensor& output_attn,
                               ov::intel_cpu::PlainTensor& output_recurrent_state,
                               uint8_t* temp_buffer,
                               const ov::intel_cpu::CpuParallelPtr& cpu_parallel,
                               const std::shared_ptr<kernel::JitKernelBase>& jit_kernel) {
    OPENVINO_ASSERT(jit_kernel, "GDN JIT kernel is not created");

    const size_t B = query.m_dims[0];
    const size_t T = query.m_dims[1];
    const size_t qk_heads = query.m_dims[2];
    const size_t K = query.m_dims[3];
    const size_t v_heads = value.m_dims[2];
    const size_t V = value.m_dims[3];
    const auto data_prc = query.m_dt;
    const size_t elem_size = ov::element::Type(data_prc).size();
    const size_t group_size = v_heads / qk_heads;
    cpu_parallel->parallel_for3d(B, v_heads, V, [&](size_t i_b, size_t i_h, size_t i_v) {
        const size_t tid = parallel_get_thread_num();
        // Use correct precision for state buffer based on data type
        uint8_t* state_buffer = temp_buffer + tid * 3 * K * elem_size;
        uint8_t* b_k = temp_buffer + tid * 3 * K * elem_size + K * elem_size;
        uint8_t* b_q = temp_buffer + tid * 3 * K * elem_size + 2 * K * elem_size;

        const size_t hk = i_h / group_size;
        auto* q_ptr = query.ptr_v(i_b, 0, hk);
        auto* k_ptr = key.ptr_v(i_b, 0, hk);
        auto* v_ptr = value.ptr_v(i_b, 0, i_h, i_v);
        auto* out_ptr = output_attn.ptr_v(i_b, 0, i_h, i_v);

        auto* gate_ptr = gate.ptr_v(i_b, 0, i_h);
        auto* beta_ptr = beta.ptr_v(i_b, 0, i_h);

        // Copy initial state in the correct precision
        if (data_prc == ov::element::f16) {
            ov::float16* init_state_f16 = reinterpret_cast<ov::float16*>(state_buffer);
            for (size_t j = 0; j < K; j++) {
                init_state_f16[j] = recurrent_state.at<ov::float16>({i_b, i_h, j, i_v});
            }
        } else if (data_prc == ov::element::bf16) {
            ov::bfloat16* init_state_bf16 = reinterpret_cast<ov::bfloat16*>(state_buffer);
            for (size_t j = 0; j < K; j++) {
                init_state_bf16[j] = recurrent_state.at<ov::bfloat16>({i_b, i_h, j, i_v});
            }
        } else {
            float* init_state_f32 = reinterpret_cast<float*>(state_buffer);
            for (size_t j = 0; j < K; j++) {
                init_state_f32[j] = recurrent_state.at<float>({i_b, i_h, j, i_v});
            }
        }

        kernel::jit_gdn_call_args args{};
        args.state = state_buffer;
        args.key_seq = reinterpret_cast<const uint8_t*>(k_ptr);
        args.query_seq = reinterpret_cast<const uint8_t*>(q_ptr);
        args.value_seq = reinterpret_cast<const uint8_t*>(v_ptr);
        args.gate_seq = reinterpret_cast<const uint8_t*>(gate_ptr);
        args.beta_seq = reinterpret_cast<const uint8_t*>(beta_ptr);
        args.t_size = T;
        args.key_query_stride = qk_heads * K;
        args.gate_beta_stride = v_heads;
        args.value_stride = v_heads * V;
        args.output_stride = v_heads * V;
        args.key_tmp = b_k;
        args.query_tmp = b_q;
        args.output_seq = reinterpret_cast<uint8_t*>(out_ptr);
        (*jit_kernel)(&args);
        // Copy final state back in the correct precision
        if (data_prc == ov::element::f16) {
            ov::float16* final_state_f16 = reinterpret_cast<ov::float16*>(state_buffer);
            for (size_t j = 0; j < K; j++) {
                output_recurrent_state.at<ov::float16>({i_b, i_h, j, i_v}) = final_state_f16[j];
            }
        } else if (data_prc == ov::element::bf16) {
            ov::bfloat16* final_state_bf16 = reinterpret_cast<ov::bfloat16*>(state_buffer);
            for (size_t j = 0; j < K; j++) {
                output_recurrent_state.at<ov::bfloat16>({i_b, i_h, j, i_v}) = final_state_bf16[j];
            }
        } else {
            float* final_state_f32 = reinterpret_cast<float*>(state_buffer);
            for (size_t j = 0; j < K; j++) {
                output_recurrent_state.at<float>({i_b, i_h, j, i_v}) = final_state_f32[j];
            }
        }
    });
}

}  // namespace
#endif

GatedDeltaNet::GatedDeltaNet(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr& context)
    : Node(op, context, NgraphShapeInferFactory(op)) {
    std::string errorMessage;
    if (!isSupportedOperation(op, errorMessage)) {
        OPENVINO_THROW_NOT_IMPLEMENTED(errorMessage);
    }
    const auto& gdn = ov::as_type_ptr<ov::op::internal::GatedDeltaNet>(op);
    m_fuse_qk_l2norm = gdn->get_fuse_qk_l2norm();
    m_q_l2_norm_eps = gdn->get_q_l2_norm_eps();
    m_k_l2_norm_eps = gdn->get_k_l2_norm_eps();
}

void GatedDeltaNet::initSupportedPrimitiveDescriptors() {
    // TODO: support other precision CVS-182464
    auto dataPrecision = getOriginalOutputPrecisionAtPort(0);
    const auto queryDims = getInputShapeAtPort(0).getDims();
    auto headSize = *(queryDims.end() - 1);
    auto implType = impl_desc_type::ref_any;
    if (ov::intel_cpu::any_of(getOriginalOutputPrecisionAtPort(0), ov::element::f16, ov::element::bf16) &&
        (mayiuse(avx512_core_bf16) || mayiuse(avx512_core_fp16)) && headSize % 32 == 0) {
        implType = impl_desc_type::jit_avx512;
    }

    std::vector<PortConfigurator> inPortConfigs;
    for (size_t i = 0; i < getParentEdges().size(); ++i) {
        inPortConfigs.emplace_back(LayoutType::ncsp, dataPrecision, getInputShapeAtPort(i), false, -1);
    }
    std::vector<PortConfigurator> outPortConfigs = {
        PortConfigurator{LayoutType::ncsp, dataPrecision, getOutputShapeAtPort(0), false, -1},
        PortConfigurator{LayoutType::ncsp, dataPrecision, getOutputShapeAtPort(1), false, -1}};
    addSupportedPrimDesc(inPortConfigs, outPortConfigs, implType);
}

void GatedDeltaNet::createPrimitive() {
    const auto precision = getOriginalOutputPrecisionAtPort(0);
    const auto queryDims = getInputShapeAtPort(0).getDims();
    auto headSize = *(queryDims.end() - 1);
    // if head_size is not multiple of 32, fallbacks to intrinsic kernel
    bool enableJit = ov::intel_cpu::any_of(precision, ov::element::f16, ov::element::bf16) &&
                     (mayiuse(avx512_core_bf16) || mayiuse(avx512_core_fp16)) && headSize % 32 == 0;
#if defined(OPENVINO_ARCH_X86_64)
    if (enableJit) {
        GatedDeltaNetKey key{precision, headSize, m_fuse_qk_l2norm, m_q_l2_norm_eps, m_k_l2_norm_eps};

        auto builder = [&](const GatedDeltaNetKey& compile_key) -> std::shared_ptr<kernel::JitKernelBase> {
            return kernel::create_gdn_jit_kernel(compile_key.precision,
                                                 compile_key.qk_head_size,
                                                 compile_key.fuse_qk_l2norm,
                                                 compile_key.q_l2_norm_eps,
                                                 compile_key.k_l2_norm_eps);
        };

        auto cache = context->getParamsCache();
        auto result = cache->getOrCreate(key, builder);
        m_gdnJitKernel = result.first;
    }
#endif

    const auto numWorkerThreads = context->getCpuParallel()->get_num_worker_threads();
    // Fallback recurrent_linear_attn uses float scratchpad; keep model/output precision unchanged.
    const auto scratchPrecision = m_gdnJitKernel ? precision : ov::element::f32;
    auto newMemDesc = std::make_shared<CpuBlockedMemoryDesc>(
        scratchPrecision,
        ov::intel_cpu::Shape{static_cast<size_t>(numWorkerThreads), 3 * headSize});
    m_tmpInpBuffer = context->getScratchPad()->createScratchPadMem(newMemDesc);
}

void GatedDeltaNet::execute([[maybe_unused]] const dnnl::stream& strm) {
    auto originalInputNumber = getOriginalInputsNumber();
    std::vector<MemoryPtr> inputs(originalInputNumber);
    std::vector<MemoryPtr> outputs(2);

    for (size_t i = 0; i < originalInputNumber; i++) {
        inputs[i] = getSrcMemoryAtPort(i);
    }

    outputs[0] = getDstMemoryAtPort(0);
    outputs[1] = getDstMemoryAtPort(1);

    PlainTensor query(inputs[0]);
    PlainTensor key(inputs[1]);
    PlainTensor value(inputs[2]);
    PlainTensor recurrent_state(inputs[3]);
    PlainTensor gate(inputs[4]);
    PlainTensor beta(inputs[5]);
    PlainTensor output_attn(outputs[0]);
    PlainTensor output_recurrent_state(outputs[1]);

    auto* temp_buffer = reinterpret_cast<uint8_t*>(m_tmpInpBuffer->getData());
#if defined(OPENVINO_ARCH_X86_64)
    if (m_gdnJitKernel) {
        recurrent_linear_attn_jit(query,
                                  key,
                                  value,
                                  recurrent_state,
                                  gate,
                                  beta,
                                  output_attn,
                                  output_recurrent_state,
                                  temp_buffer,
                                  context->getCpuParallel(),
                                  m_gdnJitKernel);
        return;
    }
#endif

    recurrent_linear_attn(query,
                          key,
                          value,
                          recurrent_state,
                          gate,
                          beta,
                          m_q_l2_norm_eps,
                          m_k_l2_norm_eps,
                          m_fuse_qk_l2norm,
                          output_attn,
                          output_recurrent_state,
                          reinterpret_cast<float*>(temp_buffer),
                          context->getCpuParallel());
}

bool GatedDeltaNet::isSupportedOperation(const std::shared_ptr<const ov::Node>& op,
                                         std::string& errorMessage) noexcept {
    if (op == nullptr || !ov::is_type<ov::op::internal::GatedDeltaNet>(op)) {
        errorMessage = "Node is not an instance of ov::op::internal::GatedDeltaNet.";
        return false;
    }
    return true;
}

}  // namespace ov::intel_cpu::node
