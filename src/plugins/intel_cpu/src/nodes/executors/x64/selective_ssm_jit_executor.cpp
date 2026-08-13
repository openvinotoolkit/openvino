// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "selective_ssm_jit_executor.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <type_traits>
#include <utility>

#include "common/utils.hpp"
#include "cpu/x64/cpu_isa_traits.hpp"
#include "memory_desc/cpu_blocked_memory_desc.h"
#include "nodes/executors/memory_arguments.hpp"
#include "nodes/kernels/selective_ssm.hpp"
#include "nodes/kernels/x64/selective_ssm_jit_kernel.hpp"
#include "openvino/core/except.hpp"
#include "openvino/core/parallel.hpp"
#include "openvino/core/type/bfloat16.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/core/type/float16.hpp"

using namespace dnnl::impl::cpu::x64;

namespace ov::intel_cpu {
namespace {

struct SelectiveSSMJitKey {
    size_t state_size;
    bool prefer_avx512;
    kernel::jit_selective_ssm_data_type data_type;

    [[nodiscard]] size_t hash() const {
        size_t seed = dnnl::impl::hash_combine(0, state_size);
        seed = dnnl::impl::hash_combine(seed, prefer_avx512);
        return dnnl::impl::hash_combine(seed, static_cast<uint8_t>(data_type));
    }

    bool operator==(const SelectiveSSMJitKey& rhs) const {
        return state_size == rhs.state_size && prefer_avx512 == rhs.prefer_avx512 && data_type == rhs.data_type;
    }
};

bool is_supported_data_precision(const ov::element::Type& precision) {
    return precision == ov::element::f32 || precision == ov::element::f16 || precision == ov::element::bf16;
}

kernel::jit_selective_ssm_data_type get_jit_data_type(const ov::element::Type& precision) {
    if (precision == ov::element::f32) {
        return kernel::jit_selective_ssm_data_type::f32;
    }
    if (precision == ov::element::f16) {
        return kernel::jit_selective_ssm_data_type::f16;
    }
    OPENVINO_ASSERT(precision == ov::element::bf16);
    return kernel::jit_selective_ssm_data_type::bf16;
}

template <typename DataT>
void copy_state_to_float(float* destination, const DataT* source, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        destination[i] = static_cast<float>(source[i]);
    }
}

template <typename DataT>
void copy_state_from_float(DataT* destination, const float* source, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        destination[i] = static_cast<DataT>(source[i]);
    }
}

template <typename DataT>
void selective_ssm_jit(const DataT* A,
                       const DataT* dt,
                       const DataT* B,
                       const DataT* x,
                       const DataT* C,
                       const DataT* recurrent_state,
                       DataT* output,
                       DataT* output_recurrent_state,
                       const node::kernel::SelectiveSSMShape& shape,
                       size_t block_head_dim,
                       float* state_scratch,
                       const CpuParallelPtr& cpu_parallel,
                       const std::shared_ptr<kernel::JitKernelBase>& jit_kernel) {
    OPENVINO_ASSERT(jit_kernel);
    const auto state_batch_stride = node::kernel::checked_size_product(
        {shape.num_heads, shape.head_dim, shape.state_size}, "recurrent state batch");
    const auto state_head_stride =
        node::kernel::checked_size_product({shape.head_dim, shape.state_size}, "state head");
    const auto scratch_stride =
        node::kernel::checked_size_product({block_head_dim, shape.state_size}, "state scratch");
    const auto heads_per_group = shape.num_heads / shape.num_groups;
    const auto p_block_count =
        shape.head_dim / block_head_dim + static_cast<size_t>(shape.head_dim % block_head_dim != 0);

    cpu_parallel
        ->parallel_for3d(shape.batch_size, shape.num_heads, p_block_count, [&](size_t batch, size_t head, size_t pb) {
            const auto p_begin = pb * block_head_dim;
            const auto p_end = p_begin + std::min(block_head_dim, shape.head_dim - p_begin);
            const auto p_count = p_end - p_begin;
            const auto group = head / heads_per_group;
            const auto state_base = batch * state_batch_stride + head * state_head_stride + p_begin * shape.state_size;

            if (shape.sequence_length == 0) {
                auto* final_state = output_recurrent_state + state_base;
                const auto* initial_state = recurrent_state + state_base;
                if (final_state != initial_state) {
                    std::memcpy(final_state, initial_state, p_count * shape.state_size * sizeof(DataT));
                }
                return;
            }

            float* working_state = nullptr;
            const float* state_source = nullptr;
            if constexpr (std::is_same_v<DataT, float>) {
                working_state = output_recurrent_state + state_base;
                state_source = recurrent_state + state_base;
            } else {
                OPENVINO_ASSERT(state_scratch);
                working_state = state_scratch + static_cast<size_t>(parallel_get_thread_num()) * scratch_stride;
                copy_state_to_float(working_state, recurrent_state + state_base, p_count * shape.state_size);
                state_source = working_state;
            }

            const float A_head = static_cast<float>(A[head]);
            auto token_head = (batch * shape.sequence_length) * shape.num_heads + head;
            auto projection_base = ((batch * shape.sequence_length) * shape.num_groups + group) * shape.state_size;
            auto x_base = token_head * shape.head_dim + p_begin;
            const auto projection_stride = shape.num_groups * shape.state_size;
            const auto x_stride = shape.num_heads * shape.head_dim;

            for (size_t token = 0; token < shape.sequence_length; ++token) {
                const float delta = static_cast<float>(dt[token_head]);
                const float decay = std::exp(A_head * delta);
                kernel::jit_selective_ssm_call_args args{state_source,
                                                         working_state,
                                                         B + projection_base,
                                                         C + projection_base,
                                                         decay,
                                                         delta,
                                                         x + x_base,
                                                         p_count,
                                                         output + x_base};
                (*jit_kernel)(&args);
                state_source = working_state;
                token_head += shape.num_heads;
                projection_base += projection_stride;
                x_base += x_stride;
            }

            if constexpr (!std::is_same_v<DataT, float>) {
                copy_state_from_float(output_recurrent_state + state_base, working_state, p_count * shape.state_size);
            }
        });
}

}  // namespace

bool SelectiveSSMJitExecutor::supports(const SelectiveSSMConfig& config) {
    if (!mayiuse(dnnl::impl::cpu::x64::avx2)) {
        return false;
    }
    const auto precision = config.descs.at(ARG_SSM_A)->getPrecision();
    if (!is_supported_data_precision(precision)) {
        return false;
    }
    for (const auto arg :
         {ARG_SSM_DT, ARG_SSM_B, ARG_SSM_X, ARG_SSM_C, ARG_SSM_STATE, ARG_SSM_OUT, ARG_SSM_OUT_STATE}) {
        if (config.descs.at(arg)->getPrecision() != precision) {
            return false;
        }
    }
    return true;
}

SelectiveSSMJitExecutor::SelectiveSSMJitExecutor(const SelectiveSSMAttrs&,
                                                 const MemoryArgs& memory,
                                                 ExecutorContext::CPtr context)
    : m_context(std::move(context)) {
    update(memory);
}

bool SelectiveSSMJitExecutor::update_scratchpad(const MemoryArgs& memory, size_t state_size, size_t block_head_dim) {
    const auto precision = memory.at(ARG_SSM_X)->getDescPtr()->getPrecision();
    if (precision == ov::element::f32) {
        return true;
    }

    if (m_state_scratch && m_cached_scratch_head_dim == block_head_dim && m_cached_state_size == state_size) {
        return true;
    }

    const auto thread_count = static_cast<size_t>(m_context->getCpuParallel()->get_num_worker_threads());
    const auto scratch_elements =
        node::kernel::checked_size_product({block_head_dim, state_size}, "JIT state scratch per worker");
    node::kernel::checked_size_product({thread_count, scratch_elements}, "JIT state scratch");
    const auto state_desc =
        std::make_shared<CpuBlockedMemoryDesc>(ov::element::f32,
                                               ov::intel_cpu::Shape{thread_count, scratch_elements});
    m_state_scratch = m_context->getScratchPad()->createScratchPadMem(state_desc);
    m_cached_scratch_head_dim = block_head_dim;
    return m_state_scratch != nullptr;
}

bool SelectiveSSMJitExecutor::update(const MemoryArgs& memory) {
    const auto& x_shape = memory.at(ARG_SSM_X)->getDescPtr()->getShape();
    const auto& B_shape = memory.at(ARG_SSM_B)->getDescPtr()->getShape();
    const auto& state_shape = memory.at(ARG_SSM_STATE)->getDescPtr()->getShape();
    if (x_shape.isDynamic() || B_shape.isDynamic() || state_shape.isDynamic()) {
        return true;
    }

    const auto& x_dims = x_shape.getStaticDims();
    const auto& B_dims = B_shape.getStaticDims();
    const auto& state_dims = state_shape.getStaticDims();
    OPENVINO_ASSERT(x_dims.size() == 4 && B_dims.size() == 4 && state_dims.size() == 4);
    const auto state_size = state_dims[3];
    const node::kernel::SelectiveSSMShape shape{x_dims[0], x_dims[1], x_dims[2], x_dims[3], B_dims[2], state_size};
    node::kernel::validate_selective_ssm_shape(shape);
    const auto thread_count = static_cast<size_t>(m_context->getCpuParallel()->get_num_worker_threads());
    m_cached_sequence_length = x_dims[1];
    const auto outer_work = node::kernel::checked_size_product({x_dims[0], x_dims[2]}, "outer work items");
    m_block_head_dim = node::kernel::get_scratch_head_dim(x_dims[3], state_size, outer_work, thread_count);
    OPENVINO_ASSERT(update_scratchpad(memory, state_size, m_block_head_dim));

    const auto precision = memory.at(ARG_SSM_X)->getDescPtr()->getPrecision();
    const auto data_type = get_jit_data_type(precision);
    const auto data_type_key = static_cast<uint8_t>(data_type);
    if (m_jit_kernel && m_decode_jit_kernel && m_cached_state_size == state_size &&
        m_cached_data_type == data_type_key) {
        return true;
    }

    const SelectiveSSMJitKey key{state_size, true, data_type};
    auto builder = [](const SelectiveSSMJitKey& compile_key) {
        return kernel::create_selective_ssm_jit_kernel(compile_key.state_size,
                                                       compile_key.prefer_avx512,
                                                       compile_key.data_type);
    };
    const auto result = m_context->getRuntimeCache()->getOrCreate(key, builder);
    m_jit_kernel = result.first;
    if (mayiuse(dnnl::impl::cpu::x64::avx512_core)) {
        const SelectiveSSMJitKey decode_key{state_size, false, data_type};
        const auto decode_result = m_context->getRuntimeCache()->getOrCreate(decode_key, builder);
        m_decode_jit_kernel = decode_result.first;
    } else {
        m_decode_jit_kernel = m_jit_kernel;
    }
    m_cached_state_size = state_size;
    m_cached_data_type = data_type_key;
    return m_jit_kernel != nullptr && m_decode_jit_kernel != nullptr;
}

void SelectiveSSMJitExecutor::execute(const MemoryArgs& memory) {
    const auto& x_dims = memory.at(ARG_SSM_X)->getStaticDims();
    const auto& B_dims = memory.at(ARG_SSM_B)->getStaticDims();
    const auto& state_dims = memory.at(ARG_SSM_STATE)->getStaticDims();
    OPENVINO_ASSERT(x_dims.size() == 4 && B_dims.size() == 4 && state_dims.size() == 4);
    const auto precision = memory.at(ARG_SSM_X)->getDescPtr()->getPrecision();
    const auto data_type_key = static_cast<uint8_t>(get_jit_data_type(precision));
    if (!m_jit_kernel || !m_decode_jit_kernel || m_cached_state_size != state_dims[3] ||
        m_cached_data_type != data_type_key || (precision != ov::element::f32 && !m_state_scratch)) {
        OPENVINO_ASSERT(update(memory));
    }

    const node::kernel::SelectiveSSMShape shape{x_dims[0], x_dims[1], x_dims[2], x_dims[3], B_dims[2], B_dims[3]};
    node::kernel::validate_selective_ssm_shape(shape);
    const auto& jit_kernel = shape.sequence_length == 1 ? m_decode_jit_kernel : m_jit_kernel;
    float* state_scratch = m_state_scratch ? m_state_scratch->getDataAs<float>() : nullptr;
#define OV_CPU_SELECTIVE_SSM_JIT_CALL(DataT)                              \
    selective_ssm_jit(memory.at(ARG_SSM_A)->getDataAs<const DataT>(),     \
                      memory.at(ARG_SSM_DT)->getDataAs<const DataT>(),    \
                      memory.at(ARG_SSM_B)->getDataAs<const DataT>(),     \
                      memory.at(ARG_SSM_X)->getDataAs<const DataT>(),     \
                      memory.at(ARG_SSM_C)->getDataAs<const DataT>(),     \
                      memory.at(ARG_SSM_STATE)->getDataAs<const DataT>(), \
                      memory.at(ARG_SSM_OUT)->getDataAs<DataT>(),         \
                      memory.at(ARG_SSM_OUT_STATE)->getDataAs<DataT>(),   \
                      shape,                                              \
                      m_block_head_dim,                                   \
                      state_scratch,                                      \
                      m_context->getCpuParallel(),                        \
                      jit_kernel)
    if (precision == ov::element::f32) {
        OV_CPU_SELECTIVE_SSM_JIT_CALL(float);
    } else if (precision == ov::element::f16) {
        OV_CPU_SELECTIVE_SSM_JIT_CALL(ov::float16);
    } else {
        OV_CPU_SELECTIVE_SSM_JIT_CALL(ov::bfloat16);
    }
#undef OV_CPU_SELECTIVE_SSM_JIT_CALL
}

impl_desc_type SelectiveSSMJitExecutor::implType() const {
    if (mayiuse(dnnl::impl::cpu::x64::avx512_core) && m_cached_sequence_length != 1) {
        return impl_desc_type::jit_avx512;
    }
    return impl_desc_type::jit_avx2;
}

}  // namespace ov::intel_cpu
