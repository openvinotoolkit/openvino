// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "nodes/executors/x64/gdn_jit_executor.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>

#include "common/utils.hpp"
#include "cpu/x64/cpu_isa_traits.hpp"
#include "cpu_parallel.hpp"
#include "memory_desc/cpu_blocked_memory_desc.h"
#include "nodes/executors/executor.hpp"
#include "nodes/executors/gated_delta_net_config.hpp"
#include "nodes/executors/memory_arguments.hpp"
#include "nodes/kernels/x64/gdn_jit_kernel.hpp"
#include "onednn/iml_type_mapper.h"
#include "openvino/core/except.hpp"
#include "openvino/core/parallel.hpp"
#include "openvino/core/type/element_type.hpp"
#include "utils/general_utils.h"
#include "utils/plain_tensor.hpp"

using namespace dnnl::impl::cpu::x64;

namespace ov::intel_cpu {
namespace {

struct GatedDeltaNetJitKey {
    ov::element::Type precision;
    size_t qk_head_size;
    size_t v_tile;
    float q_l2_norm_eps;
    float k_l2_norm_eps;
    bool fuse_qk_l2norm;

    [[nodiscard]] size_t hash() const {
        size_t seed = 0;
        seed = dnnl::impl::hash_combine(seed, precision.hash());
        seed = dnnl::impl::hash_combine(seed, qk_head_size);
        seed = dnnl::impl::hash_combine(seed, v_tile);
        seed = dnnl::impl::hash_combine(seed, q_l2_norm_eps);
        seed = dnnl::impl::hash_combine(seed, k_l2_norm_eps);
        seed = dnnl::impl::hash_combine(seed, fuse_qk_l2norm);
        return seed;
    }

    bool operator==(const GatedDeltaNetJitKey& rhs) const {
        return precision == rhs.precision && qk_head_size == rhs.qk_head_size && v_tile == rhs.v_tile &&
               q_l2_norm_eps == rhs.q_l2_norm_eps && k_l2_norm_eps == rhs.k_l2_norm_eps &&
               fuse_qk_l2norm == rhs.fuse_qk_l2norm;
    }
};

size_t getJitVTile(const ov::element::Type& precision, const size_t defaultTile) {
    return precision == ov::element::f32 ? 1 : defaultTile;
}

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
    OPENVINO_ASSERT(ov::intel_cpu::any_of(data_prc, ov::element::f16, ov::element::bf16, ov::element::f32),
                    "GDN JIT supports only f16/bf16/f32 state copy path");
    const size_t group_size = v_heads / qk_heads;
    OPENVINO_ASSERT(V % gdn_jit_v_tile == 0, "GDN JIT requires V divisible by ", gdn_jit_v_tile, ", got V=", V);
    const size_t v_tiles = V / gdn_jit_v_tile;
    const size_t state_tile_size = gdn_jit_v_tile * K * elem_size;
    const size_t thread_buffer_size = (gdn_jit_v_tile + 2) * K * elem_size;

    cpu_parallel->parallel_for3d(B, v_heads, v_tiles, [&](size_t i_b, size_t i_h, size_t i_v_tile) {
        const size_t tid = parallel_get_thread_num();
        const size_t i_v_begin = i_v_tile * gdn_jit_v_tile;

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

        if (data_prc == ov::element::f32) {
            auto* init_state_f32 = reinterpret_cast<float*>(state_buffer);
            const auto* recurrent_state_f32 =
                reinterpret_cast<const float*>(recurrent_state.ptr_v(i_b, i_h, 0, i_v_begin));
            for (size_t j = 0; j < K; j++) {
                const auto* src_row = recurrent_state_f32 + j * recurrent_state_stride_k;
                for (size_t v_idx = 0; v_idx < gdn_jit_v_tile; v_idx++) {
                    init_state_f32[v_idx * K + j] = src_row[v_idx * recurrent_state_stride_v];
                }
            }
        } else {
            auto* init_state_u16 = reinterpret_cast<uint16_t*>(state_buffer);
            const auto* recurrent_state_u16 =
                reinterpret_cast<const uint16_t*>(recurrent_state.ptr_v(i_b, i_h, 0, i_v_begin));
            for (size_t j = 0; j < K; j++) {
                const auto* src_row = recurrent_state_u16 + j * recurrent_state_stride_k;
                for (size_t v_idx = 0; v_idx < gdn_jit_v_tile; v_idx++) {
                    init_state_u16[v_idx * K + j] = src_row[v_idx * recurrent_state_stride_v];
                }
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

        if (data_prc == ov::element::f32) {
            const auto* final_state_f32 = reinterpret_cast<const float*>(state_buffer);
            auto* output_state_f32 = reinterpret_cast<float*>(output_recurrent_state.ptr_v(i_b, i_h, 0, i_v_begin));
            for (size_t j = 0; j < K; j++) {
                auto* dst_row = output_state_f32 + j * output_state_stride_k;
                for (size_t v_idx = 0; v_idx < gdn_jit_v_tile; v_idx++) {
                    dst_row[v_idx * output_state_stride_v] = final_state_f32[v_idx * K + j];
                }
            }
        } else {
            const auto* final_state_u16 = reinterpret_cast<const uint16_t*>(state_buffer);
            auto* output_state_u16 = reinterpret_cast<uint16_t*>(output_recurrent_state.ptr_v(i_b, i_h, 0, i_v_begin));
            for (size_t j = 0; j < K; j++) {
                auto* dst_row = output_state_u16 + j * output_state_stride_k;
                for (size_t v_idx = 0; v_idx < gdn_jit_v_tile; v_idx++) {
                    dst_row[v_idx * output_state_stride_v] = final_state_u16[v_idx * K + j];
                }
            }
        }
    });
}

}  // namespace

bool GdnJitExecutor::supports(const GatedDeltaNetConfig& config) {
    const auto precision = config.descs.at(ARG_GDN_QUERY)->getPrecision();
    if (precision == ov::element::f32) {
        return mayiuse(dnnl::impl::cpu::x64::avx512_core) || mayiuse(dnnl::impl::cpu::x64::avx2);
    }

    if (precision == ov::element::f16) {
        return mayiuse(dnnl::impl::cpu::x64::avx512_core_fp16);
    }
    if (precision == ov::element::bf16) {
        return mayiuse(dnnl::impl::cpu::x64::avx512_core_bf16);
    }

    return false;
}

GdnJitExecutor::GdnJitExecutor(const GatedDeltaNetAttrs& attrs, const MemoryArgs& memory, ExecutorContext::CPtr context)
    : m_attrs(attrs),
      m_context(std::move(context)) {
    update(memory);
}

bool GdnJitExecutor::updateKernelAndScratchpad(const MemoryArgs& memory) {
    const auto precision = memory.at(ARG_GDN_QUERY)->getDescPtr()->getPrecision();
    const auto& queryShape = memory.at(ARG_GDN_QUERY)->getDescPtr()->getShape();
    const auto& queryDims = queryShape.getStaticDims();
    const size_t headSize = queryDims.back();
    const size_t jitVTile = getJitVTile(precision, m_attrs.jit_v_tile);

    GatedDeltaNetJitKey key{precision,
                            headSize,
                            jitVTile,
                            m_attrs.q_l2_norm_eps,
                            m_attrs.k_l2_norm_eps,
                            m_attrs.fuse_qk_l2norm};

    auto builder = [](const GatedDeltaNetJitKey& compile_key) -> std::shared_ptr<kernel::JitKernelBase> {
        return kernel::create_gdn_jit_kernel(compile_key.precision,
                                             compile_key.qk_head_size,
                                             compile_key.v_tile,
                                             compile_key.fuse_qk_l2norm,
                                             compile_key.q_l2_norm_eps,
                                             compile_key.k_l2_norm_eps);
    };

    auto cache = m_context->getRuntimeCache();
    auto result = cache->getOrCreate(key, builder);
    m_jitKernel = result.first;
    OPENVINO_ASSERT(m_jitKernel, "Failed to create GatedDeltaNet JIT kernel");

    const bool needRecreateScratchpad = m_tmpInpBuffer == nullptr || m_cachedPrecision != precision ||
                                        m_cachedHeadSize != headSize || m_cachedVTile != jitVTile;
    if (needRecreateScratchpad) {
        const auto numWorkerThreads = m_context->getCpuParallel()->get_num_worker_threads();
        auto newMemDesc = std::make_shared<CpuBlockedMemoryDesc>(
            precision,
            ov::intel_cpu::Shape{static_cast<size_t>(numWorkerThreads), (jitVTile + 2) * headSize});
        m_tmpInpBuffer = m_context->getScratchPad()->createScratchPadMem(newMemDesc);
    }

    m_cachedPrecision = precision;
    m_cachedHeadSize = headSize;
    m_cachedVTile = jitVTile;

    return m_jitKernel != nullptr && m_tmpInpBuffer != nullptr;
}

bool GdnJitExecutor::update(const MemoryArgs& memory) {
    return updateKernelAndScratchpad(memory);
}

void GdnJitExecutor::execute(const MemoryArgs& memory) {
    OPENVINO_ASSERT(m_jitKernel != nullptr, "GatedDeltaNet JIT executor kernel is not initialized");
    OPENVINO_ASSERT(m_tmpInpBuffer != nullptr, "GatedDeltaNet JIT executor scratchpad is not initialized");

    PlainTensor query(memory.at(ARG_GDN_QUERY));
    PlainTensor key(memory.at(ARG_GDN_KEY));
    PlainTensor value(memory.at(ARG_GDN_VALUE));
    PlainTensor recurrent_state(memory.at(ARG_GDN_STATE));
    PlainTensor gate(memory.at(ARG_GDN_GATE));
    PlainTensor beta(memory.at(ARG_GDN_BETA));
    PlainTensor output_attn(memory.at(ARG_GDN_OUT_ATTN));
    PlainTensor output_recurrent_state(memory.at(ARG_GDN_OUT_STATE));

    auto* temp_buffer = reinterpret_cast<uint8_t*>(m_tmpInpBuffer->getData());
    recurrent_linear_attn_jit(query,
                              key,
                              value,
                              recurrent_state,
                              gate,
                              beta,
                              output_attn,
                              output_recurrent_state,
                              temp_buffer,
                              m_context->getCpuParallel(),
                              m_jitKernel,
                              m_cachedVTile);
}

impl_desc_type GdnJitExecutor::implType() const {
    if (mayiuse(dnnl::impl::cpu::x64::avx512_core)) {
        return impl_desc_type::jit_avx512;
    }
    return impl_desc_type::jit_avx2;
}

}  // namespace ov::intel_cpu
