// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gated_delta_net.h"

#include <cstddef>
#include <cstdint>
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
#include "utils/plain_tensor.hpp"
#if defined(OPENVINO_ARCH_X86_64)
#    include "cpu_parallel.hpp"
#    include "kernels/x64/gdn_jit_kernel.hpp"
using namespace dnnl::impl::cpu::x64;
#endif

using namespace ov::Extensions::Cpu;
using namespace ov::Extensions::Cpu::XARCH;

namespace ov::intel_cpu::node {

#if defined(OPENVINO_ARCH_X86_64)
namespace {
struct GatedDeltaNetKey {
    ov::element::Type precision;
    size_t qk_head_size;
    size_t v_tile;
    bool fuse_qk_l2norm;
    float q_l2_norm_eps;
    float k_l2_norm_eps;

    [[nodiscard]] size_t hash() const {
        size_t seed = 0;
        seed = dnnl::impl::hash_combine(seed, precision.hash());
        seed = dnnl::impl::hash_combine(seed, qk_head_size);
        seed = dnnl::impl::hash_combine(seed, v_tile);
        seed = dnnl::impl::hash_combine(seed, fuse_qk_l2norm);
        seed = dnnl::impl::hash_combine(seed, q_l2_norm_eps);
        seed = dnnl::impl::hash_combine(seed, k_l2_norm_eps);
        return seed;
    }

    bool operator==(const GatedDeltaNetKey& rhs) const {
        return precision == rhs.precision && qk_head_size == rhs.qk_head_size && v_tile == rhs.v_tile &&
               fuse_qk_l2norm == rhs.fuse_qk_l2norm && q_l2_norm_eps == rhs.q_l2_norm_eps &&
               k_l2_norm_eps == rhs.k_l2_norm_eps;
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
                               const std::shared_ptr<kernel::JitKernelBase>& jit_kernel,
                               const size_t gdn_jit_v_tile) {
    OPENVINO_ASSERT(jit_kernel, "GDN JIT kernel is not created");

    const size_t B = query.m_dims[0];
    const size_t T = query.m_dims[1];
    const size_t qk_heads = query.m_dims[2];
    const size_t K = query.m_dims[3];
    const size_t v_heads = value.m_dims[2];
    const size_t V = value.m_dims[3];
    const auto data_prc = query.m_dt;
    const size_t elem_size = ov::element::Type(data_prc).size();
    OPENVINO_ASSERT(ov::intel_cpu::any_of(data_prc, ov::element::f16, ov::element::bf16),
                    "GDN JIT supports only f16/bf16 state copy path");
    const size_t group_size = v_heads / qk_heads;
    OPENVINO_ASSERT(V % gdn_jit_v_tile == 0, "GDN JIT requires V divisible by ", gdn_jit_v_tile, ", got V=", V);
    const size_t v_tiles = V / gdn_jit_v_tile;
    const size_t state_tile_size = gdn_jit_v_tile * K * elem_size;
    const size_t thread_buffer_size = (gdn_jit_v_tile + 2) * K * elem_size;
    cpu_parallel->parallel_for3d(B, v_heads, v_tiles, [&](size_t i_b, size_t i_h, size_t i_v_tile) {
        const size_t tid = parallel_get_thread_num();
        const size_t i_v_begin = i_v_tile * gdn_jit_v_tile;

        // Per-thread layout: [state tile][key tmp][query tmp]
        uint8_t* state_buffer = temp_buffer + tid * thread_buffer_size;
        uint8_t* b_k = state_buffer + state_tile_size;
        uint8_t* b_q = b_k + K * elem_size;

        const size_t hk = i_h / group_size;
        auto* q_ptr = query.ptr_v(i_b, 0, hk);
        auto* k_ptr = key.ptr_v(i_b, 0, hk);
        auto* v_ptr = value.ptr_v(i_b, 0, i_h, i_v_begin);
        auto* out_ptr = output_attn.ptr_v(i_b, 0, i_h, i_v_begin);

        auto* gate_ptr = gate.ptr_v(i_b, 0, i_h);
        auto* beta_ptr = beta.ptr_v(i_b, 0, i_h);

        const size_t recurrent_state_stride_k = recurrent_state.stride(2);
        const size_t recurrent_state_stride_v = recurrent_state.stride(3);
        const size_t output_state_stride_k = output_recurrent_state.stride(2);
        const size_t output_state_stride_v = output_recurrent_state.stride(3);

        // JIT path stores state in 2-byte elements (f16/bf16).
        auto* init_state_u16 = reinterpret_cast<uint16_t*>(state_buffer);
        auto* recurrent_state_u16 = reinterpret_cast<const uint16_t*>(recurrent_state.ptr_v(i_b, i_h, 0, i_v_begin));
        for (size_t j = 0; j < K; j++) {
            const auto* src_row = recurrent_state_u16 + j * recurrent_state_stride_k;
            for (size_t v_idx = 0; v_idx < gdn_jit_v_tile; v_idx++) {
                init_state_u16[v_idx * K + j] = src_row[v_idx * recurrent_state_stride_v];
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

        // Copy final state tile back (2-byte elements: f16/bf16).
        auto* final_state_u16 = reinterpret_cast<const uint16_t*>(state_buffer);
        auto* output_state_u16 = reinterpret_cast<uint16_t*>(output_recurrent_state.ptr_v(i_b, i_h, 0, i_v_begin));
        for (size_t j = 0; j < K; j++) {
            auto* dst_row = output_state_u16 + j * output_state_stride_k;
            for (size_t v_idx = 0; v_idx < gdn_jit_v_tile; v_idx++) {
                dst_row[v_idx * output_state_stride_v] = final_state_u16[v_idx * K + j];
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
    auto implType = impl_desc_type::ref_any;
#if defined(OPENVINO_ARCH_X86_64)
    const auto queryDims = getInputShapeAtPort(0).getDims();
    auto headSize = *(queryDims.end() - 1);
    if (ov::intel_cpu::any_of(getOriginalOutputPrecisionAtPort(0), ov::element::f16, ov::element::bf16) &&
        (mayiuse(avx512_core_bf16) || mayiuse(avx512_core_fp16)) && headSize % 32 == 0) {
        implType = impl_desc_type::jit_avx512;
        m_enableJit = true;
    }
#endif

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
    const auto queryDims = getInputShapeAtPort(0).getDims();
    auto headSize = *(queryDims.end() - 1);
    size_t scratchRows = 3;
    auto scratchPrecision = ov::element::f32;
    // if head_size is not multiple of 32, fallbacks to intrinsic kernel
#if defined(OPENVINO_ARCH_X86_64)
    if (m_enableJit) {
        const auto precision = getOriginalOutputPrecisionAtPort(0);
        GatedDeltaNetKey key{precision, headSize, m_gdnJitVTile, m_fuse_qk_l2norm, m_q_l2_norm_eps, m_k_l2_norm_eps};

        auto builder = [&](const GatedDeltaNetKey& compile_key) -> std::shared_ptr<kernel::JitKernelBase> {
            return kernel::create_gdn_jit_kernel(compile_key.precision,
                                                 compile_key.qk_head_size,
                                                 compile_key.v_tile,
                                                 compile_key.fuse_qk_l2norm,
                                                 compile_key.q_l2_norm_eps,
                                                 compile_key.k_l2_norm_eps);
        };

        auto cache = context->getParamsCache();
        auto result = cache->getOrCreate(key, builder);
        m_gdnJitKernel = result.first;
        if (m_gdnJitKernel) {
            scratchPrecision = precision;
            scratchRows = m_gdnJitVTile + 2;
        }
    }
#endif

    const auto numWorkerThreads = context->getCpuParallel()->get_num_worker_threads();
    auto newMemDesc = std::make_shared<CpuBlockedMemoryDesc>(
        scratchPrecision,
        ov::intel_cpu::Shape{static_cast<size_t>(numWorkerThreads), scratchRows * headSize});
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
        OPENVINO_ASSERT(value.m_dims[3] % m_gdnJitVTile == 0,
                        "GDN JIT requires V divisible by ",
                        m_gdnJitVTile,
                        ", got V=",
                        value.m_dims[3]);

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
                                  m_gdnJitKernel,
                                  m_gdnJitVTile);
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
