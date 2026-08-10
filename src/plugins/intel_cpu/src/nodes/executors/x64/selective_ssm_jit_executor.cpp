// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "selective_ssm_jit_executor.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <memory>
#include <utility>

#include "common/utils.hpp"
#include "cpu/x64/cpu_isa_traits.hpp"
#include "nodes/executors/memory_arguments.hpp"
#include "nodes/kernels/selective_ssm.hpp"
#include "nodes/kernels/x64/selective_ssm_jit_kernel.hpp"
#include "openvino/core/except.hpp"
#include "openvino/core/type/element_type.hpp"

using namespace dnnl::impl::cpu::x64;

namespace ov::intel_cpu {
namespace {

struct SelectiveSSMJitKey {
    size_t state_size;
    bool prefer_avx512;

    [[nodiscard]] size_t hash() const {
        size_t seed = dnnl::impl::hash_combine(0, state_size);
        return dnnl::impl::hash_combine(seed, prefer_avx512);
    }

    bool operator==(const SelectiveSSMJitKey& rhs) const {
        return state_size == rhs.state_size && prefer_avx512 == rhs.prefer_avx512;
    }
};

void selective_ssm_jit_f32(const float* A,
                           const float* dt,
                           const float* B,
                           const float* x,
                           const float* C,
                           const float* recurrent_state,
                           float* output,
                           float* output_recurrent_state,
                           const node::kernel::SelectiveSSMShape& shape,
                           size_t block_head_dim,
                           const CpuParallelPtr& cpu_parallel,
                           const std::shared_ptr<kernel::JitKernelBase>& jit_kernel) {
    OPENVINO_ASSERT(jit_kernel);
    const auto state_batch_stride = shape.num_heads * shape.head_dim * shape.state_size;
    const auto state_head_stride = shape.head_dim * shape.state_size;
    const auto heads_per_group = shape.num_heads / shape.num_groups;
    const auto p_block_count = (shape.head_dim + block_head_dim - 1) / block_head_dim;

    cpu_parallel
        ->parallel_for3d(shape.batch_size, shape.num_heads, p_block_count, [&](size_t batch, size_t head, size_t pb) {
            const auto p_begin = pb * block_head_dim;
            const auto p_end = std::min(p_begin + block_head_dim, shape.head_dim);
            const auto p_count = p_end - p_begin;
            const auto group = head / heads_per_group;
            const auto state_base = batch * state_batch_stride + head * state_head_stride + p_begin * shape.state_size;

            if (shape.sequence_length == 0) {
                auto* final_state = output_recurrent_state + state_base;
                const auto* initial_state = recurrent_state + state_base;
                if (final_state != initial_state) {
                    std::memcpy(final_state, initial_state, p_count * shape.state_size * sizeof(float));
                }
                return;
            }

            const float A_head = A[head];
            auto token_head = (batch * shape.sequence_length) * shape.num_heads + head;
            auto projection_base = ((batch * shape.sequence_length) * shape.num_groups + group) * shape.state_size;
            auto x_base = token_head * shape.head_dim + p_begin;
            const auto projection_stride = shape.num_groups * shape.state_size;
            const auto x_stride = shape.num_heads * shape.head_dim;

            auto execute_token = [&](const float* state_source) {
                const float delta = dt[token_head];
                const float decay = std::exp(A_head * delta);
                kernel::jit_selective_ssm_call_args args{state_source,
                                                         output_recurrent_state + state_base,
                                                         B + projection_base,
                                                         C + projection_base,
                                                         decay,
                                                         delta,
                                                         x + x_base,
                                                         p_count,
                                                         output + x_base};
                (*jit_kernel)(&args);
            };

            execute_token(recurrent_state + state_base);
            token_head += shape.num_heads;
            projection_base += projection_stride;
            x_base += x_stride;
            for (size_t token = 1; token < shape.sequence_length; ++token) {
                execute_token(output_recurrent_state + state_base);
                token_head += shape.num_heads;
                projection_base += projection_stride;
                x_base += x_stride;
            }
        });
}

}  // namespace

bool SelectiveSSMJitExecutor::supports(const SelectiveSSMConfig& config) {
    if (!mayiuse(dnnl::impl::cpu::x64::avx2)) {
        return false;
    }
    for (const auto arg :
         {ARG_SSM_A, ARG_SSM_DT, ARG_SSM_B, ARG_SSM_X, ARG_SSM_C, ARG_SSM_STATE, ARG_SSM_OUT, ARG_SSM_OUT_STATE}) {
        if (config.descs.at(arg)->getPrecision() != ov::element::f32) {
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

bool SelectiveSSMJitExecutor::update(const MemoryArgs& memory) {
    const auto& x_shape = memory.at(ARG_SSM_X)->getDescPtr()->getShape();
    const auto& state_shape = memory.at(ARG_SSM_STATE)->getDescPtr()->getShape();
    if (x_shape.isDynamic() || state_shape.isDynamic()) {
        return true;
    }

    const auto& x_dims = x_shape.getStaticDims();
    const auto& state_dims = state_shape.getStaticDims();
    OPENVINO_ASSERT(x_dims.size() == 4 && state_dims.size() == 4);
    const auto state_size = state_dims[3];
    const auto thread_count = static_cast<size_t>(m_context->getCpuParallel()->get_num_worker_threads());
    m_cached_sequence_length = x_dims[1];
    m_block_head_dim = node::kernel::get_scratch_head_dim(x_dims[3], state_size, x_dims[0] * x_dims[2], thread_count);

    if (m_jit_kernel && m_decode_jit_kernel && m_cached_state_size == state_size) {
        return true;
    }

    const SelectiveSSMJitKey key{state_size, true};
    auto builder = [](const SelectiveSSMJitKey& compile_key) {
        return kernel::create_selective_ssm_jit_kernel(compile_key.state_size, compile_key.prefer_avx512);
    };
    const auto result = m_context->getRuntimeCache()->getOrCreate(key, builder);
    m_jit_kernel = result.first;
    if (mayiuse(dnnl::impl::cpu::x64::avx512_core)) {
        const SelectiveSSMJitKey decode_key{state_size, false};
        const auto decode_result = m_context->getRuntimeCache()->getOrCreate(decode_key, builder);
        m_decode_jit_kernel = decode_result.first;
    } else {
        m_decode_jit_kernel = m_jit_kernel;
    }
    m_cached_state_size = state_size;
    return m_jit_kernel != nullptr && m_decode_jit_kernel != nullptr;
}

void SelectiveSSMJitExecutor::execute(const MemoryArgs& memory) {
    const auto& x_dims = memory.at(ARG_SSM_X)->getStaticDims();
    const auto& B_dims = memory.at(ARG_SSM_B)->getStaticDims();
    const auto& state_dims = memory.at(ARG_SSM_STATE)->getStaticDims();
    OPENVINO_ASSERT(x_dims.size() == 4 && B_dims.size() == 4 && state_dims.size() == 4);
    if (!m_jit_kernel || m_cached_state_size != state_dims[3]) {
        OPENVINO_ASSERT(update(memory));
    }

    const node::kernel::SelectiveSSMShape shape{x_dims[0], x_dims[1], x_dims[2], x_dims[3], B_dims[2], B_dims[3]};
    const auto& jit_kernel = shape.sequence_length == 1 ? m_decode_jit_kernel : m_jit_kernel;
    selective_ssm_jit_f32(memory.at(ARG_SSM_A)->getDataAs<const float>(),
                          memory.at(ARG_SSM_DT)->getDataAs<const float>(),
                          memory.at(ARG_SSM_B)->getDataAs<const float>(),
                          memory.at(ARG_SSM_X)->getDataAs<const float>(),
                          memory.at(ARG_SSM_C)->getDataAs<const float>(),
                          memory.at(ARG_SSM_STATE)->getDataAs<const float>(),
                          memory.at(ARG_SSM_OUT)->getDataAs<float>(),
                          memory.at(ARG_SSM_OUT_STATE)->getDataAs<float>(),
                          shape,
                          m_block_head_dim,
                          m_context->getCpuParallel(),
                          jit_kernel);
}

impl_desc_type SelectiveSSMJitExecutor::implType() const {
    if (mayiuse(dnnl::impl::cpu::x64::avx512_core) && m_cached_sequence_length != 1) {
        return impl_desc_type::jit_avx512;
    }
    return impl_desc_type::jit_avx2;
}

}  // namespace ov::intel_cpu
