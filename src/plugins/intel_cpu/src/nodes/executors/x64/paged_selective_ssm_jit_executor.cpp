// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "paged_selective_ssm_jit_executor.hpp"

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

struct PagedSelectiveSSMJitKey {
    size_t state_size;
    bool prefer_avx512;
    kernel::jit_selective_ssm_data_type data_type;

    [[nodiscard]] size_t hash() const {
        size_t seed = dnnl::impl::hash_combine(0, state_size);
        seed = dnnl::impl::hash_combine(seed, prefer_avx512);
        return dnnl::impl::hash_combine(seed, static_cast<uint8_t>(data_type));
    }

    bool operator==(const PagedSelectiveSSMJitKey& rhs) const {
        return state_size == rhs.state_size && prefer_avx512 == rhs.prefer_avx512 && data_type == rhs.data_type;
    }
};

bool is_supported_data_precision(const ov::element::Type& precision) {
    return precision == ov::element::f32 || precision == ov::element::f16 || precision == ov::element::bf16;
}

bool is_supported_index_precision(const ov::element::Type& precision) {
    return precision == ov::element::i32 || precision == ov::element::i64;
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
    if constexpr (std::is_same_v<DataT, float>) {
        std::memcpy(destination, source, count * sizeof(float));
    } else {
        for (size_t i = 0; i < count; ++i) {
            destination[i] = static_cast<float>(source[i]);
        }
    }
}

template <typename DataT>
void copy_state_from_float(DataT* destination, const float* source, size_t count) {
    if constexpr (std::is_same_v<DataT, float>) {
        std::memcpy(destination, source, count * sizeof(float));
    } else {
        for (size_t i = 0; i < count; ++i) {
            destination[i] = static_cast<DataT>(source[i]);
        }
    }
}

template <typename DataT, typename IndexT>
void paged_selective_ssm_jit(const DataT* A,
                             const DataT* dt,
                             const DataT* B,
                             const DataT* x,
                             const DataT* C,
                             DataT* recurrent_state_table,
                             const IndexT* subsequence_begins,
                             const IndexT* block_indices,
                             const IndexT* block_indices_begins,
                             const IndexT* num_processed_tokens,
                             const IndexT* cache_interval,
                             DataT* output,
                             const node::kernel::PagedSelectiveSSMShape& shape,
                             float* state_scratch,
                             size_t scratch_head_dim,
                             const CpuParallelPtr& cpu_parallel,
                             const std::shared_ptr<kernel::JitKernelBase>& jit_kernel,
                             const std::shared_ptr<kernel::JitKernelBase>& decode_jit_kernel) {
    const auto block_stride =
        node::kernel::checked_size_product({shape.num_heads, shape.head_dim, shape.state_size}, "state block");
    const auto head_stride =
        node::kernel::checked_size_product({shape.head_dim, shape.state_size}, "state head");
    const auto scratch_stride =
        node::kernel::checked_size_product({scratch_head_dim, shape.state_size}, "state scratch");
    const auto heads_per_group = shape.num_heads / shape.num_groups;
    const auto p_block_count =
        shape.head_dim / scratch_head_dim + static_cast<size_t>(shape.head_dim % scratch_head_dim != 0);

    cpu_parallel->parallel_for3d(
        shape.sequence_count,
        shape.num_heads,
        p_block_count,
        [&](size_t sequence, size_t head, size_t pb) {
            const auto token_begin = static_cast<size_t>(subsequence_begins[sequence]);
            const auto token_end = static_cast<size_t>(subsequence_begins[sequence + 1]);
            if (token_begin == token_end) {
                return;
            }

            const auto p_begin = pb * scratch_head_dim;
            const auto p_end = p_begin + std::min(scratch_head_dim, shape.head_dim - p_begin);
            const auto p_count = p_end - p_begin;
            const auto group = head / heads_per_group;
            const auto logical_block_begin = static_cast<size_t>(block_indices_begins[sequence]);
            const auto read_block = static_cast<size_t>(block_indices[logical_block_begin]);
            const auto state_slice = head * head_stride + p_begin * shape.state_size;
            const auto* initial_state = recurrent_state_table + read_block * block_stride + state_slice;
            const float A_head = static_cast<float>(A[head]);
            const auto interval = cache_interval[sequence];
            const bool cache_enabled = interval > 0;
            const uint64_t positive_interval = cache_enabled ? static_cast<uint64_t>(interval) : 1;
            const uint64_t cache_offset =
                cache_enabled ? static_cast<uint64_t>(num_processed_tokens[sequence]) % positive_interval : 0;
            auto token_head = token_begin * shape.num_heads + head;
            auto projection_base = (token_begin * shape.num_groups + group) * shape.state_size;
            auto x_base = token_head * shape.head_dim + p_begin;
            const auto projection_stride = shape.num_groups * shape.state_size;
            const auto x_stride = shape.num_heads * shape.head_dim;
            const auto& selected_kernel = token_end - token_begin == 1 ? decode_jit_kernel : jit_kernel;

            if constexpr (std::is_same_v<DataT, float>) {
                if (token_end - token_begin == 1 && cache_enabled) {
                    const float delta = dt[token_head];
                    const float decay = std::exp(A_head * delta);
                    const auto write_block = static_cast<size_t>(block_indices[logical_block_begin + 1]);
                    auto* state_destination = recurrent_state_table + write_block * block_stride + state_slice;
                    kernel::jit_selective_ssm_call_args args{initial_state,
                                                             state_destination,
                                                             B + projection_base,
                                                             C + projection_base,
                                                             decay,
                                                             delta,
                                                             x + x_base,
                                                             p_count,
                                                             output + x_base};
                    (*decode_jit_kernel)(&args);
                    return;
                }
            }

            auto* local_state = state_scratch + static_cast<size_t>(parallel_get_thread_num()) * scratch_stride;
            copy_state_to_float(local_state, initial_state, p_count * shape.state_size);
            uint64_t tokens_until_boundary = cache_enabled ? positive_interval - cache_offset : 0;
            size_t write_slot = 1;

            for (size_t token = token_begin; token < token_end; ++token) {
                const float delta = static_cast<float>(dt[token_head]);
                const float decay = std::exp(A_head * delta);
                kernel::jit_selective_ssm_call_args args{local_state,
                                                         local_state,
                                                         B + projection_base,
                                                         C + projection_base,
                                                         decay,
                                                         delta,
                                                         x + x_base,
                                                         p_count,
                                                         output + x_base};
                (*selected_kernel)(&args);

                if (cache_enabled) {
                    const bool is_boundary = --tokens_until_boundary == 0;
                    const bool is_last = token + 1 == token_end;
                    if (is_boundary || is_last) {
                        const auto write_block = static_cast<size_t>(block_indices[logical_block_begin + write_slot++]);
                        auto* snapshot = recurrent_state_table + write_block * block_stride + state_slice;
                        copy_state_from_float(snapshot, local_state, p_count * shape.state_size);
                    }
                    if (is_boundary) {
                        tokens_until_boundary = positive_interval;
                    }
                }
                token_head += shape.num_heads;
                projection_base += projection_stride;
                x_base += x_stride;
            }
        });
}

}  // namespace

bool PagedSelectiveSSMJitExecutor::supports(const PagedSelectiveSSMConfig& config) {
    if (!mayiuse(dnnl::impl::cpu::x64::avx2)) {
        return false;
    }
    const auto data_precision = config.descs.at(ARG_PAGED_SSM_A)->getPrecision();
    if (!is_supported_data_precision(data_precision)) {
        return false;
    }
    for (const auto arg : {ARG_PAGED_SSM_DT,
                           ARG_PAGED_SSM_B,
                           ARG_PAGED_SSM_X,
                           ARG_PAGED_SSM_C,
                           ARG_PAGED_SSM_STATE,
                           ARG_PAGED_SSM_OUT}) {
        if (config.descs.at(arg)->getPrecision() != data_precision) {
            return false;
        }
    }

    const auto index_precision = config.descs.at(ARG_PAGED_SSM_SUBSEQUENCE_BEGINS)->getPrecision();
    if (!is_supported_index_precision(index_precision)) {
        return false;
    }
    for (const auto arg : {ARG_PAGED_SSM_BLOCK_INDICES,
                           ARG_PAGED_SSM_BLOCK_INDICES_BEGINS,
                           ARG_PAGED_SSM_NUM_PROCESSED_TOKENS,
                           ARG_PAGED_SSM_CACHE_INTERVAL}) {
        if (config.descs.at(arg)->getPrecision() != index_precision) {
            return false;
        }
    }
    return true;
}

PagedSelectiveSSMJitExecutor::PagedSelectiveSSMJitExecutor(const PagedSelectiveSSMAttrs&,
                                                           const MemoryArgs& memory,
                                                           ExecutorContext::CPtr context)
    : m_context(std::move(context)) {
    update(memory);
}

bool PagedSelectiveSSMJitExecutor::update_scratchpad(const MemoryArgs& memory) {
    const auto& x_shape = memory.at(ARG_PAGED_SSM_X)->getDescPtr()->getShape();
    const auto& B_shape = memory.at(ARG_PAGED_SSM_B)->getDescPtr()->getShape();
    const auto& state_shape = memory.at(ARG_PAGED_SSM_STATE)->getDescPtr()->getShape();
    if (x_shape.isDynamic() || B_shape.isDynamic() || state_shape.isDynamic()) {
        return true;
    }
    const auto& x_dims = x_shape.getStaticDims();
    const auto& B_dims = B_shape.getStaticDims();
    const auto& state_dims = state_shape.getStaticDims();
    OPENVINO_ASSERT(x_dims.size() == 3 && B_dims.size() == 3 && state_dims.size() == 4);
    const auto head_dim = x_dims[2];
    const auto state_size = state_dims[3];
    const auto physical_blocks = state_dims[0];
    const node::kernel::PagedSelectiveSSMShape shape{x_dims[0],
                                                     x_dims[1],
                                                     head_dim,
                                                     B_dims[1],
                                                     state_size,
                                                     physical_blocks,
                                                     0,
                                                     0};
    node::kernel::validate_paged_selective_ssm_shape(shape);
    const auto thread_count = static_cast<size_t>(m_context->getCpuParallel()->get_num_worker_threads());
    const auto scratch_head_dim = node::kernel::get_scratch_head_dim(head_dim, state_size, x_dims[1], thread_count);
    const auto scratch_elements =
        node::kernel::checked_size_product({scratch_head_dim, state_size}, "JIT state scratch per worker");
    node::kernel::checked_size_product({thread_count, scratch_elements}, "JIT state scratch");
    if (m_state_scratch && m_block_owners && m_scratch_head_dim == scratch_head_dim &&
        m_cached_state_size == state_size && m_cached_physical_blocks == physical_blocks) {
        return true;
    }

    const auto state_desc =
        std::make_shared<CpuBlockedMemoryDesc>(ov::element::f32,
                                               ov::intel_cpu::Shape{thread_count, scratch_elements});
    const auto owner_desc =
        std::make_shared<CpuBlockedMemoryDesc>(ov::element::i32,
                                               ov::intel_cpu::Shape{std::max(size_t{1}, physical_blocks)});
    m_state_scratch = m_context->getScratchPad()->createScratchPadMem(state_desc);
    m_block_owners = m_context->getScratchPad()->createScratchPadMem(owner_desc);
    m_scratch_head_dim = scratch_head_dim;
    m_cached_physical_blocks = physical_blocks;
    return m_state_scratch != nullptr && m_block_owners != nullptr;
}

bool PagedSelectiveSSMJitExecutor::update(const MemoryArgs& memory) {
    const auto& x_shape = memory.at(ARG_PAGED_SSM_X)->getDescPtr()->getShape();
    const auto& state_shape = memory.at(ARG_PAGED_SSM_STATE)->getDescPtr()->getShape();
    const auto& subsequence_shape = memory.at(ARG_PAGED_SSM_SUBSEQUENCE_BEGINS)->getDescPtr()->getShape();
    if (x_shape.isDynamic() || state_shape.isDynamic() || subsequence_shape.isDynamic()) {
        return true;
    }
    OPENVINO_ASSERT(update_scratchpad(memory));

    const auto& x_dims = x_shape.getStaticDims();
    const auto& state_dims = state_shape.getStaticDims();
    const auto& subsequence_dims = subsequence_shape.getStaticDims();
    const auto state_size = state_dims[3];
    m_cached_token_count = x_dims[0];
    m_cached_sequence_count = subsequence_dims[0] - 1;
    const auto precision = memory.at(ARG_PAGED_SSM_X)->getDescPtr()->getPrecision();
    const auto data_type = get_jit_data_type(precision);
    const auto data_type_key = static_cast<uint8_t>(data_type);
    if (m_jit_kernel && m_decode_jit_kernel && m_cached_state_size == state_size &&
        m_cached_data_type == data_type_key) {
        return true;
    }

    const PagedSelectiveSSMJitKey key{state_size, true, data_type};
    auto builder = [](const PagedSelectiveSSMJitKey& compile_key) {
        return kernel::create_selective_ssm_jit_kernel(compile_key.state_size,
                                                       compile_key.prefer_avx512,
                                                       compile_key.data_type);
    };
    const auto result = m_context->getRuntimeCache()->getOrCreate(key, builder);
    m_jit_kernel = result.first;
    if (mayiuse(dnnl::impl::cpu::x64::avx512_core)) {
        const PagedSelectiveSSMJitKey decode_key{state_size, false, data_type};
        const auto decode_result = m_context->getRuntimeCache()->getOrCreate(decode_key, builder);
        m_decode_jit_kernel = decode_result.first;
    } else {
        m_decode_jit_kernel = m_jit_kernel;
    }
    m_cached_state_size = state_size;
    m_cached_data_type = data_type_key;
    return m_jit_kernel != nullptr && m_decode_jit_kernel != nullptr;
}

void PagedSelectiveSSMJitExecutor::execute(const MemoryArgs& memory) {
    const auto& B_dims = memory.at(ARG_PAGED_SSM_B)->getStaticDims();
    const auto& x_dims = memory.at(ARG_PAGED_SSM_X)->getStaticDims();
    const auto& state_dims = memory.at(ARG_PAGED_SSM_STATE)->getStaticDims();
    const auto& subsequence_dims = memory.at(ARG_PAGED_SSM_SUBSEQUENCE_BEGINS)->getStaticDims();
    const auto& block_indices_dims = memory.at(ARG_PAGED_SSM_BLOCK_INDICES)->getStaticDims();
    const auto& block_begins_dims = memory.at(ARG_PAGED_SSM_BLOCK_INDICES_BEGINS)->getStaticDims();
    const auto& processed_dims = memory.at(ARG_PAGED_SSM_NUM_PROCESSED_TOKENS)->getStaticDims();
    const auto& interval_dims = memory.at(ARG_PAGED_SSM_CACHE_INTERVAL)->getStaticDims();
    OPENVINO_ASSERT(B_dims.size() == 3 && x_dims.size() == 3 && state_dims.size() == 4);
    OPENVINO_ASSERT(subsequence_dims.size() == 1 && subsequence_dims[0] >= 1 && block_indices_dims.size() == 1);
    const auto sequence_count = subsequence_dims[0] - 1;
    OPENVINO_ASSERT(block_begins_dims.size() == 1 && processed_dims.size() == 1 && interval_dims.size() == 1 &&
                        block_begins_dims[0] == sequence_count + 1 && processed_dims[0] == sequence_count &&
                        interval_dims[0] == sequence_count,
                    "PagedSelectiveSSM metadata tensor lengths are inconsistent.");
    const auto precision = memory.at(ARG_PAGED_SSM_X)->getDescPtr()->getPrecision();
    const auto data_type_key = static_cast<uint8_t>(get_jit_data_type(precision));
    if (!m_jit_kernel || !m_decode_jit_kernel || !m_state_scratch || !m_block_owners ||
        m_cached_state_size != state_dims[3] || m_cached_physical_blocks != state_dims[0] ||
        m_cached_data_type != data_type_key) {
        OPENVINO_ASSERT(update(memory));
    }

    const node::kernel::PagedSelectiveSSMShape shape{x_dims[0],
                                                     x_dims[1],
                                                     x_dims[2],
                                                     B_dims[1],
                                                     B_dims[2],
                                                     state_dims[0],
                                                     block_indices_dims[0],
                                                     sequence_count};
    node::kernel::validate_paged_selective_ssm_shape(shape);
    const auto index_precision = memory.at(ARG_PAGED_SSM_SUBSEQUENCE_BEGINS)->getDescPtr()->getPrecision();
    node::kernel::validate_paged_selective_ssm_metadata(memory.at(ARG_PAGED_SSM_SUBSEQUENCE_BEGINS)->getData(),
                                                        memory.at(ARG_PAGED_SSM_BLOCK_INDICES)->getData(),
                                                        memory.at(ARG_PAGED_SSM_BLOCK_INDICES_BEGINS)->getData(),
                                                        memory.at(ARG_PAGED_SSM_NUM_PROCESSED_TOKENS)->getData(),
                                                        memory.at(ARG_PAGED_SSM_CACHE_INTERVAL)->getData(),
                                                        shape,
                                                        index_precision,
                                                        m_block_owners->getDataAs<int32_t>());

#define OV_CPU_PAGED_SSM_JIT_CALL(DataT, IndexT)                                                      \
    paged_selective_ssm_jit(memory.at(ARG_PAGED_SSM_A)->getDataAs<const DataT>(),                     \
                            memory.at(ARG_PAGED_SSM_DT)->getDataAs<const DataT>(),                    \
                            memory.at(ARG_PAGED_SSM_B)->getDataAs<const DataT>(),                     \
                            memory.at(ARG_PAGED_SSM_X)->getDataAs<const DataT>(),                     \
                            memory.at(ARG_PAGED_SSM_C)->getDataAs<const DataT>(),                     \
                            memory.at(ARG_PAGED_SSM_STATE)->getDataAs<DataT>(),                       \
                            memory.at(ARG_PAGED_SSM_SUBSEQUENCE_BEGINS)->getDataAs<const IndexT>(),   \
                            memory.at(ARG_PAGED_SSM_BLOCK_INDICES)->getDataAs<const IndexT>(),        \
                            memory.at(ARG_PAGED_SSM_BLOCK_INDICES_BEGINS)->getDataAs<const IndexT>(), \
                            memory.at(ARG_PAGED_SSM_NUM_PROCESSED_TOKENS)->getDataAs<const IndexT>(), \
                            memory.at(ARG_PAGED_SSM_CACHE_INTERVAL)->getDataAs<const IndexT>(),       \
                            memory.at(ARG_PAGED_SSM_OUT)->getDataAs<DataT>(),                         \
                            shape,                                                                    \
                            m_state_scratch->getDataAs<float>(),                                      \
                            m_scratch_head_dim,                                                       \
                            m_context->getCpuParallel(),                                              \
                            m_jit_kernel,                                                             \
                            m_decode_jit_kernel)
#define OV_CPU_PAGED_SSM_JIT_DISPATCH_INDEX(DataT) \
    if (index_precision == ov::element::i32) {     \
        OV_CPU_PAGED_SSM_JIT_CALL(DataT, int32_t); \
    } else {                                       \
        OV_CPU_PAGED_SSM_JIT_CALL(DataT, int64_t); \
    }
    if (precision == ov::element::f32) {
        OV_CPU_PAGED_SSM_JIT_DISPATCH_INDEX(float);
    } else if (precision == ov::element::f16) {
        OV_CPU_PAGED_SSM_JIT_DISPATCH_INDEX(ov::float16);
    } else {
        OV_CPU_PAGED_SSM_JIT_DISPATCH_INDEX(ov::bfloat16);
    }
#undef OV_CPU_PAGED_SSM_JIT_DISPATCH_INDEX
#undef OV_CPU_PAGED_SSM_JIT_CALL
}

impl_desc_type PagedSelectiveSSMJitExecutor::implType() const {
    if (mayiuse(dnnl::impl::cpu::x64::avx512_core) && m_cached_token_count > m_cached_sequence_count) {
        return impl_desc_type::jit_avx512;
    }
    return impl_desc_type::jit_avx2;
}

}  // namespace ov::intel_cpu
