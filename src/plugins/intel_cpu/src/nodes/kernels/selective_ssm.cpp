// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "selective_ssm.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <limits>
#include <type_traits>

#include "cpu_parallel.hpp"
#include "nodes/kernels/scaled_attn/common.hpp"
#include "openvino/core/except.hpp"
#include "openvino/core/parallel.hpp"
#include "openvino/core/type/bfloat16.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/core/type/float16.hpp"
#include "openvino/util/math_util.hpp"
#include "utils/general_utils.h"
#include "utils/plain_tensor.hpp"

namespace ov::intel_cpu::node::kernel {
namespace {

// Bound the portable executor's FP32 recurrent-state scratch independently of the CPU cache topology.
// Machine-specific executors may use a cache-aware tiling policy instead.
constexpr size_t max_scratch_bytes_per_worker = 32 * 1024;

template <typename DstT, typename SrcT>
inline void copy_convert(DstT* dst, const SrcT* src, size_t count) {
    if constexpr (std::is_same_v<DstT, SrcT>) {
        if (dst != src) {
            std::memcpy(dst, src, count * sizeof(DstT));
        }
    } else {
        ov::Extensions::Cpu::XARCH::cvt_copy(dst, const_cast<SrcT*>(src), 1, count, count, count);
    }
}

template <typename ProjectionT>
inline float update_state_and_reduce(float* state,
                                     const ProjectionT* input_projection,
                                     const ProjectionT* output_projection,
                                     float decay,
                                     float input_scale,
                                     size_t state_size) {
    for (size_t n = 0; n < state_size; ++n) {
        state[n] = state[n] * decay + static_cast<float>(input_projection[n]) * input_scale;
    }

    if constexpr (std::is_same_v<ProjectionT, float>) {
        return ov::Extensions::Cpu::XARCH::dot_product(state,
                                                       output_projection,
                                                       state_size,
                                                       nullptr,
                                                       nullptr,
                                                       nullptr,
                                                       0);
    }

    float result = 0.F;
    for (size_t n = 0; n < state_size; ++n) {
        result += state[n] * static_cast<float>(output_projection[n]);
    }
    return result;
}

template <typename DataT, typename ProjectionT>
void selective_ssm_typed(const DataT* A,
                         const DataT* dt,
                         const ProjectionT* B,
                         const DataT* x,
                         const ProjectionT* C,
                         const DataT* recurrent_state,
                         DataT* output,
                         DataT* output_recurrent_state,
                         const SelectiveSSMShape& shape,
                         float* state_scratch,
                         size_t scratch_head_dim,
                         const CpuParallelPtr& cpu_parallel) {
    OPENVINO_ASSERT(shape.num_groups > 0, "SelectiveSSM requires a positive number of groups.");
    const auto heads_per_group = shape.num_heads / shape.num_groups;
    OPENVINO_ASSERT(heads_per_group > 0 && heads_per_group * shape.num_groups == shape.num_heads,
                    "SelectiveSSM requires the number of groups to evenly divide the number of heads.");
    OPENVINO_ASSERT(scratch_head_dim > 0, "SelectiveSSM scratch head dimension must be positive.");

    const auto batch_state_stride =
        checked_size_product({shape.num_heads, shape.head_dim, shape.state_size}, "recurrent state batch");
    const auto head_state_stride = checked_size_product({shape.head_dim, shape.state_size}, "recurrent state head");
    const auto scratch_stride = checked_size_product({scratch_head_dim, shape.state_size}, "state scratch");
    const auto p_block_count = ov::util::ceil_div(shape.head_dim, scratch_head_dim);

    cpu_parallel
        ->parallel_for3d(shape.batch_size, shape.num_heads, p_block_count, [&](size_t batch, size_t head, size_t pb) {
            const auto p_begin = pb * scratch_head_dim;
            const auto p_end = p_begin + std::min(scratch_head_dim, shape.head_dim - p_begin);
            const auto p_count = p_end - p_begin;
            const auto group = head / heads_per_group;
            const auto state_base = batch * batch_state_stride + head * head_state_stride + p_begin * shape.state_size;
            float* local_state = nullptr;
            if constexpr (std::is_same_v<DataT, float>) {
                // FP32 can use the final state as its working buffer. Lower precisions retain FP32 scratch to avoid
                // quantizing the recurrent state at every token.
                local_state = output_recurrent_state + state_base;
            } else {
                local_state = state_scratch + static_cast<size_t>(parallel_get_thread_num()) * scratch_stride;
            }

            const auto state_elements = p_count * shape.state_size;
            copy_convert(local_state, recurrent_state + state_base, state_elements);

            const auto a_head = static_cast<float>(A[head]);
            auto token_head = (batch * shape.sequence_length) * shape.num_heads + head;
            auto projection_base = ((batch * shape.sequence_length) * shape.num_groups + group) * shape.state_size;
            auto x_base = token_head * shape.head_dim + p_begin;
            const auto projection_stride = shape.num_groups * shape.state_size;
            const auto x_stride = shape.num_heads * shape.head_dim;

            for (size_t token = 0; token < shape.sequence_length; ++token) {
                const auto delta = static_cast<float>(dt[token_head]);
                const float decay = std::exp(a_head * delta);
                const auto* input_projection = B + projection_base;
                const auto* output_projection = C + projection_base;
                for (size_t p = 0; p < p_count; ++p) {
                    auto* state = local_state + p * shape.state_size;
                    const float input_scale = static_cast<float>(x[x_base + p]) * delta;
                    const float result = update_state_and_reduce(state,
                                                                 input_projection,
                                                                 output_projection,
                                                                 decay,
                                                                 input_scale,
                                                                 shape.state_size);
                    output[x_base + p] = static_cast<DataT>(result);
                }

                token_head += shape.num_heads;
                projection_base += projection_stride;
                x_base += x_stride;
            }

            if constexpr (!std::is_same_v<DataT, float>) {
                copy_convert(output_recurrent_state + state_base, local_state, state_elements);
            }
        });
}

struct PagedMetadata {
    ov::intel_cpu::PlainTensor subsequence_begins;
    ov::intel_cpu::PlainTensor block_indices;
    ov::intel_cpu::PlainTensor block_indices_begins;
    ov::intel_cpu::PlainTensor num_processed_tokens;
    ov::intel_cpu::PlainTensor cache_interval;
};

int64_t index_at(const ov::intel_cpu::PlainTensor& tensor, size_t index) {
    return tensor.get_precision() == ov::element::i64 ? tensor.ptr<int64_t>()[index] : tensor.ptr<int32_t>()[index];
}

ov::intel_cpu::PlainTensor make_index_tensor(const void* data, size_t count, const ov::element::Type& index_precision) {
    ov::intel_cpu::PlainTensor tensor;
    tensor.resize({count}, index_precision.size(), index_precision, const_cast<void*>(data));
    return tensor;
}

PagedMetadata make_paged_metadata(const void* subsequence_begins,
                                  const void* block_indices,
                                  const void* block_indices_begins,
                                  const void* num_processed_tokens,
                                  const void* cache_interval,
                                  const ov::element::Type& index_precision,
                                  const PagedSelectiveSSMShape& shape) {
    OPENVINO_ASSERT(any_of(index_precision, ov::element::i32, ov::element::i64),
                    "PagedSelectiveSSM supports only i32/i64 metadata, got ",
                    index_precision,
                    ".");
    const auto sequence_offsets_count = checked_size_sum({shape.sequence_count, size_t{1}}, "metadata offsets");
    return {make_index_tensor(subsequence_begins, sequence_offsets_count, index_precision),
            make_index_tensor(block_indices, shape.logical_block_count, index_precision),
            make_index_tensor(block_indices_begins, sequence_offsets_count, index_precision),
            make_index_tensor(num_processed_tokens, shape.sequence_count, index_precision),
            make_index_tensor(cache_interval, shape.sequence_count, index_precision)};
}

size_t checked_index(int64_t value, size_t limit, const char* name, size_t position) {
    OPENVINO_ASSERT(value >= 0, "PagedSelectiveSSM: ", name, "[", position, "] must be non-negative, got ", value, ".");
    const auto result = static_cast<uint64_t>(value);
    OPENVINO_ASSERT(result < limit,
                    "PagedSelectiveSSM: ",
                    name,
                    "[",
                    position,
                    "] is out of range: ",
                    value,
                    " not in [0, ",
                    limit,
                    ").");
    return static_cast<size_t>(result);
}

void validate_paged_metadata(const PagedMetadata& metadata,
                             const PagedSelectiveSSMShape& shape,
                             int32_t* block_owners) {
    // Metadata controls pointer arithmetic and parallel cache writes and may change on every execution. Validate it
    // here, limiting the per-sequence work below to the read block and the writable blocks the kernel will consume.
    const auto& subsequence_begins = metadata.subsequence_begins;
    const auto& block_indices = metadata.block_indices;
    const auto& block_indices_begins = metadata.block_indices_begins;
    const auto& num_processed_tokens = metadata.num_processed_tokens;
    const auto& cache_interval = metadata.cache_interval;
    OPENVINO_ASSERT(shape.sequence_count <= static_cast<size_t>(std::numeric_limits<int32_t>::max()),
                    "PagedSelectiveSSM supports at most INT32_MAX sequences.");
    OPENVINO_ASSERT(shape.physical_block_count == 0 || block_owners != nullptr,
                    "PagedSelectiveSSM requires block-owner scratch for a non-empty state table.");

    OPENVINO_ASSERT(index_at(subsequence_begins, 0) == 0,
                    "PagedSelectiveSSM: subsequence_begins[0] must be 0, got ",
                    index_at(subsequence_begins, 0),
                    ".");
    OPENVINO_ASSERT(index_at(block_indices_begins, 0) == 0,
                    "PagedSelectiveSSM: block_indices_begins[0] must be 0, got ",
                    index_at(block_indices_begins, 0),
                    ".");
    const auto final_token_offset = index_at(subsequence_begins, shape.sequence_count);
    OPENVINO_ASSERT(static_cast<uint64_t>(final_token_offset) == shape.token_count,
                    "PagedSelectiveSSM: the last subsequence offset must equal token_count (",
                    shape.token_count,
                    "), got ",
                    final_token_offset,
                    ".");
    const auto final_block_offset = index_at(block_indices_begins, shape.sequence_count);
    OPENVINO_ASSERT(static_cast<uint64_t>(final_block_offset) == shape.logical_block_count,
                    "PagedSelectiveSSM: the last block offset must equal logical_block_count (",
                    shape.logical_block_count,
                    "), got ",
                    final_block_offset,
                    ".");

    if (shape.physical_block_count > 0) {
        std::fill(block_owners, block_owners + shape.physical_block_count, int32_t{-1});
    }

    // Validate only the metadata consumed by each sequence and register its writable blocks in the same pass.
    for (size_t s = 0; s < shape.sequence_count; ++s) {
        const auto token_begin = index_at(subsequence_begins, s);
        const auto token_end = index_at(subsequence_begins, s + 1);
        const auto block_begin_value = index_at(block_indices_begins, s);
        const auto block_end_value = index_at(block_indices_begins, s + 1);
        const auto processed_tokens = index_at(num_processed_tokens, s);
        OPENVINO_ASSERT(token_begin >= 0 && token_end >= token_begin,
                        "PagedSelectiveSSM: subsequence_begins must be non-negative and non-decreasing at sequence ",
                        s,
                        ".");
        OPENVINO_ASSERT(block_begin_value >= 0 && block_end_value >= block_begin_value,
                        "PagedSelectiveSSM: block_indices_begins must be non-negative and non-decreasing at sequence ",
                        s,
                        ".");
        OPENVINO_ASSERT(processed_tokens >= 0,
                        "PagedSelectiveSSM: num_processed_tokens[",
                        s,
                        "] must be non-negative, got ",
                        processed_tokens,
                        ".");

        const auto token_count = static_cast<uint64_t>(token_end - token_begin);
        if (token_count == 0) {
            continue;
        }

        const auto block_begin = static_cast<uint64_t>(block_begin_value);
        const auto block_end = static_cast<uint64_t>(block_end_value);
        OPENVINO_ASSERT(block_end > block_begin,
                        "PagedSelectiveSSM: non-empty sequence ",
                        s,
                        " requires a read block.");

        const auto interval = index_at(cache_interval, s);
        if (interval <= 0) {
            continue;
        }
        const auto positive_interval = static_cast<uint64_t>(interval);
        const auto offset = static_cast<uint64_t>(processed_tokens) % positive_interval;
        OPENVINO_ASSERT(token_count <= std::numeric_limits<uint64_t>::max() - offset,
                        "PagedSelectiveSSM: token count overflow at sequence ",
                        s,
                        ".");
        const auto cached_token_count = offset + token_count;
        const auto write_count = (cached_token_count - 1) / positive_interval + 1;
        OPENVINO_ASSERT(block_end - block_begin - 1 >= write_count,
                        "PagedSelectiveSSM: sequence ",
                        s,
                        " requires ",
                        write_count,
                        " writable logical blocks after the read block, got ",
                        block_end - block_begin - 1,
                        ".");

        for (uint64_t slot = 1; slot <= write_count; ++slot) {
            const auto logical = static_cast<size_t>(block_begin + slot);
            const auto physical =
                checked_index(index_at(block_indices, logical), shape.physical_block_count, "block_indices", logical);
            OPENVINO_ASSERT(block_owners[physical] == -1,
                            "PagedSelectiveSSM: physical block ",
                            physical,
                            " is written more than once (previous sequence ",
                            block_owners[physical],
                            ", current sequence ",
                            s,
                            ").");
            block_owners[physical] = static_cast<int32_t>(s);
        }
    }

    // A read may alias this sequence's first write (the documented in-place case), but it
    // must not race with another sequence's writer. Read/read sharing is harmless.
    for (size_t s = 0; s < shape.sequence_count; ++s) {
        if (index_at(subsequence_begins, s + 1) == index_at(subsequence_begins, s)) {
            continue;
        }
        const auto logical = static_cast<size_t>(index_at(block_indices_begins, s));
        const auto physical =
            checked_index(index_at(block_indices, logical), shape.physical_block_count, "block_indices", logical);
        const auto owner = block_owners[physical];
        bool aliases_first_write = false;
        if (owner == static_cast<int32_t>(s)) {
            const auto first_write_logical = logical + 1;
            const auto first_write_physical = checked_index(index_at(block_indices, first_write_logical),
                                                            shape.physical_block_count,
                                                            "block_indices",
                                                            first_write_logical);
            aliases_first_write = first_write_physical == physical;
        }
        OPENVINO_ASSERT(owner == -1 || aliases_first_write,
                        "PagedSelectiveSSM: sequence ",
                        s,
                        " reads physical block ",
                        physical,
                        " while sequence ",
                        owner,
                        " writes it; a read may alias only the same sequence's first write.");
    }
}

template <typename DataT, typename StateT, typename ProjectionT>
void paged_selective_ssm_typed(const DataT* A,
                               const DataT* dt,
                               const ProjectionT* B,
                               const DataT* x,
                               const ProjectionT* C,
                               StateT* recurrent_state_table,
                               const PagedMetadata& metadata,
                               DataT* output,
                               const PagedSelectiveSSMShape& shape,
                               float* state_scratch,
                               size_t scratch_head_dim,
                               const CpuParallelPtr& cpu_parallel) {
    OPENVINO_ASSERT(shape.num_groups > 0, "PagedSelectiveSSM requires a positive number of groups.");
    const auto heads_per_group = shape.num_heads / shape.num_groups;
    OPENVINO_ASSERT(heads_per_group > 0 && heads_per_group * shape.num_groups == shape.num_heads,
                    "PagedSelectiveSSM requires the number of groups to evenly divide the number of heads.");
    OPENVINO_ASSERT(scratch_head_dim > 0, "PagedSelectiveSSM scratch head dimension must be positive.");

    const auto& subsequence_begins = metadata.subsequence_begins;
    const auto& block_indices = metadata.block_indices;
    const auto& block_indices_begins = metadata.block_indices_begins;
    const auto& num_processed_tokens = metadata.num_processed_tokens;
    const auto& cache_interval = metadata.cache_interval;

    const auto block_stride = checked_size_product({shape.num_heads, shape.head_dim, shape.state_size}, "state block");
    const auto head_stride = checked_size_product({shape.head_dim, shape.state_size}, "state head");
    const auto scratch_stride = checked_size_product({scratch_head_dim, shape.state_size}, "state scratch");
    const auto p_block_count = ov::util::ceil_div(shape.head_dim, scratch_head_dim);

    cpu_parallel->parallel_for3d(
        shape.sequence_count,
        shape.num_heads,
        p_block_count,
        [&](size_t sequence, size_t head, size_t pb) {
            const auto token_begin = static_cast<size_t>(index_at(subsequence_begins, sequence));
            const auto token_end = static_cast<size_t>(index_at(subsequence_begins, sequence + 1));
            if (token_begin == token_end) {
                return;
            }

            const auto p_begin = pb * scratch_head_dim;
            const auto p_end = p_begin + std::min(scratch_head_dim, shape.head_dim - p_begin);
            const auto p_count = p_end - p_begin;
            const auto group = head / heads_per_group;
            const auto logical_block_begin = static_cast<size_t>(index_at(block_indices_begins, sequence));
            const auto read_block = static_cast<size_t>(index_at(block_indices, logical_block_begin));
            const auto state_slice = head * head_stride + p_begin * shape.state_size;
            const auto* initial_state = recurrent_state_table + read_block * block_stride + state_slice;
            const auto a_head = static_cast<float>(A[head]);
            const auto interval = index_at(cache_interval, sequence);
            const bool cache_enabled = interval > 0;
            auto token_head = token_begin * shape.num_heads + head;
            auto projection_base = (token_begin * shape.num_groups + group) * shape.state_size;
            auto x_base = token_head * shape.head_dim + p_begin;
            const auto projection_stride = shape.num_groups * shape.state_size;
            const auto x_stride = shape.num_heads * shape.head_dim;
            auto* local_state = state_scratch + static_cast<size_t>(parallel_get_thread_num()) * scratch_stride;
            const auto state_elements = p_count * shape.state_size;
            copy_convert(local_state, initial_state, state_elements);
            const uint64_t positive_interval = cache_enabled ? static_cast<uint64_t>(interval) : 1;
            const uint64_t cache_offset =
                cache_enabled ? static_cast<uint64_t>(index_at(num_processed_tokens, sequence)) % positive_interval : 0;

            for (size_t token = token_begin; token < token_end; ++token) {
                const auto delta = static_cast<float>(dt[token_head]);
                const float decay = std::exp(a_head * delta);
                const auto* input_projection = B + projection_base;
                const auto* output_projection = C + projection_base;
                for (size_t p = 0; p < p_count; ++p) {
                    auto* state = local_state + p * shape.state_size;
                    const float input_scale = static_cast<float>(x[x_base + p]) * delta;
                    const float result = update_state_and_reduce(state,
                                                                 input_projection,
                                                                 output_projection,
                                                                 decay,
                                                                 input_scale,
                                                                 shape.state_size);
                    output[x_base + p] = static_cast<DataT>(result);
                }

                const auto processed_tokens = static_cast<uint64_t>((token - token_begin) + 1);
                const auto cached_tokens = cache_offset + processed_tokens;
                const bool interval_hit = cache_enabled && cached_tokens % positive_interval == 0;
                const bool is_last = token + 1 == token_end;
                if (cache_enabled && (interval_hit || is_last)) {
                    const auto write_slot = 1 + (cached_tokens - 1) / positive_interval;
                    const auto write_block =
                        static_cast<size_t>(index_at(block_indices, logical_block_begin + write_slot));
                    auto* snapshot = recurrent_state_table + write_block * block_stride + state_slice;
                    copy_convert(snapshot, local_state, state_elements);
                }

                token_head += shape.num_heads;
                projection_base += projection_stride;
                x_base += x_stride;
            }
        });
}

template <typename DataT>
void dispatch_selective_projection(const void* A,
                                   const void* dt,
                                   const void* B,
                                   const void* x,
                                   const void* C,
                                   const void* recurrent_state,
                                   void* output,
                                   void* output_recurrent_state,
                                   const SelectiveSSMShape& shape,
                                   float* state_scratch,
                                   size_t scratch_head_dim,
                                   const CpuParallelPtr& cpu_parallel,
                                   const float* converted_B,
                                   const float* converted_C) {
    const auto dispatch = [&](const auto* B_data, const auto* C_data) {
        selective_ssm_typed(static_cast<const DataT*>(A),
                            static_cast<const DataT*>(dt),
                            B_data,
                            static_cast<const DataT*>(x),
                            C_data,
                            static_cast<const DataT*>(recurrent_state),
                            static_cast<DataT*>(output),
                            static_cast<DataT*>(output_recurrent_state),
                            shape,
                            state_scratch,
                            scratch_head_dim,
                            cpu_parallel);
    };
    if (converted_B != nullptr) {
        dispatch(converted_B, converted_C);
    } else {
        dispatch(static_cast<const DataT*>(B), static_cast<const DataT*>(C));
    }
}

template <typename DataT, typename StateT>
void dispatch_paged_projection(const void* A,
                               const void* dt,
                               const void* B,
                               const void* x,
                               const void* C,
                               void* recurrent_state_table,
                               const PagedMetadata& metadata,
                               void* output,
                               const PagedSelectiveSSMShape& shape,
                               float* state_scratch,
                               size_t scratch_head_dim,
                               const CpuParallelPtr& cpu_parallel,
                               const float* converted_B,
                               const float* converted_C) {
    const auto dispatch = [&](const auto* B_data, const auto* C_data) {
        paged_selective_ssm_typed(static_cast<const DataT*>(A),
                                  static_cast<const DataT*>(dt),
                                  B_data,
                                  static_cast<const DataT*>(x),
                                  C_data,
                                  static_cast<StateT*>(recurrent_state_table),
                                  metadata,
                                  static_cast<DataT*>(output),
                                  shape,
                                  state_scratch,
                                  scratch_head_dim,
                                  cpu_parallel);
    };
    if (converted_B != nullptr) {
        dispatch(converted_B, converted_C);
    } else {
        dispatch(static_cast<const DataT*>(B), static_cast<const DataT*>(C));
    }
}

struct PagedSelectiveSSMCallArgs {
    const void* A;
    const void* dt;
    const void* B;
    const void* x;
    const void* C;
    void* recurrent_state_table;
    const PagedMetadata& metadata;
    void* output;
    const PagedSelectiveSSMShape& shape;
    float* state_scratch;
    size_t scratch_head_dim;
    const CpuParallelPtr& cpu_parallel;
    const float* converted_B;
    const float* converted_C;
};

template <typename DataT, typename StateT>
void dispatch_paged_typed(const PagedSelectiveSSMCallArgs& args) {
    dispatch_paged_projection<DataT, StateT>(args.A,
                                             args.dt,
                                             args.B,
                                             args.x,
                                             args.C,
                                             args.recurrent_state_table,
                                             args.metadata,
                                             args.output,
                                             args.shape,
                                             args.state_scratch,
                                             args.scratch_head_dim,
                                             args.cpu_parallel,
                                             args.converted_B,
                                             args.converted_C);
}

template <typename DataT>
void dispatch_paged_state_type(const PagedSelectiveSSMCallArgs& args, const ov::element::Type& state_precision) {
    switch (state_precision) {
    case ov::element::f32:
        dispatch_paged_typed<DataT, float>(args);
        return;
    case ov::element::f16:
        dispatch_paged_typed<DataT, ov::float16>(args);
        return;
    case ov::element::bf16:
        dispatch_paged_typed<DataT, ov::bfloat16>(args);
        return;
    default:
        OPENVINO_THROW("PagedSelectiveSSM supports only f32/f16/bf16 state, got ", state_precision, ".");
    }
}

}  // namespace

size_t checked_size_product(std::initializer_list<size_t> dimensions, const char* tensor_name) {
    if (std::find(dimensions.begin(), dimensions.end(), size_t{0}) != dimensions.end()) {
        return 0;
    }

    size_t result = 1;
    for (const auto dimension : dimensions) {
        size_t product = 0;
        OPENVINO_ASSERT(!ov::util::mul_overflow(result, dimension, product),
                        "SelectiveSSM size overflow while calculating ",
                        tensor_name,
                        ".");
        result = product;
    }
    return result;
}

size_t checked_size_sum(std::initializer_list<size_t> values, const char* buffer_name) {
    size_t result = 0;
    for (const auto value : values) {
        OPENVINO_ASSERT(result <= std::numeric_limits<size_t>::max() - value,
                        "SelectiveSSM size overflow while calculating ",
                        buffer_name,
                        ".");
        result += value;
    }
    return result;
}

size_t get_scratch_head_dim(size_t head_dim, size_t state_size, size_t outer_work_items, size_t thread_count) {
    OPENVINO_ASSERT(state_size > 0);
    if (head_dim == 0) {
        return 1;
    }
    const auto scratch_elements = max_scratch_bytes_per_worker / sizeof(float);
    const auto scratch_limited = std::max(size_t{1}, std::min(head_dim, scratch_elements / state_size));
    const auto outer_work = std::max(size_t{1}, outer_work_items);
    const auto workers = std::max(size_t{1}, thread_count);
    const auto blocks_for_parallelism = ov::util::ceil_div(workers, outer_work);
    const auto parallelism_limited = ov::util::ceil_div(head_dim, blocks_for_parallelism);
    return std::max(size_t{1}, std::min(scratch_limited, parallelism_limited));
}

void selective_ssm(const void* A,
                   const void* dt,
                   const void* B,
                   const void* x,
                   const void* C,
                   const void* recurrent_state,
                   void* output,
                   void* output_recurrent_state,
                   const SelectiveSSMShape& shape,
                   const ov::element::Type& precision,
                   float* state_scratch,
                   size_t scratch_head_dim,
                   const CpuParallelPtr& cpu_parallel,
                   const float* converted_B,
                   const float* converted_C) {
    OPENVINO_ASSERT(scratch_head_dim > 0 && state_scratch != nullptr);
    OPENVINO_ASSERT(cpu_parallel != nullptr, "SelectiveSSM requires a CPU parallel executor.");
    OPENVINO_ASSERT((converted_B == nullptr) == (converted_C == nullptr));
#define OV_CPU_SSM_CALL(DataT)                                   \
    dispatch_selective_projection<DataT>(A,                      \
                                         dt,                     \
                                         B,                      \
                                         x,                      \
                                         C,                      \
                                         recurrent_state,        \
                                         output,                 \
                                         output_recurrent_state, \
                                         shape,                  \
                                         state_scratch,          \
                                         scratch_head_dim,       \
                                         cpu_parallel,           \
                                         converted_B,            \
                                         converted_C)
    if (precision == ov::element::f32) {
        OV_CPU_SSM_CALL(float);
    } else if (precision == ov::element::f16) {
        OV_CPU_SSM_CALL(ov::float16);
    } else if (precision == ov::element::bf16) {
        OV_CPU_SSM_CALL(ov::bfloat16);
    } else {
        OPENVINO_THROW("SelectiveSSM supports only f32/f16/bf16, got ", precision, ".");
    }
#undef OV_CPU_SSM_CALL
}

void paged_selective_ssm(const void* A,
                         const void* dt,
                         const void* B,
                         const void* x,
                         const void* C,
                         void* recurrent_state_table,
                         const void* subsequence_begins,
                         const void* block_indices,
                         const void* block_indices_begins,
                         const void* num_processed_tokens,
                         const void* cache_interval,
                         void* output,
                         const PagedSelectiveSSMShape& shape,
                         const ov::element::Type& data_precision,
                         const ov::element::Type& state_precision,
                         const ov::element::Type& index_precision,
                         float* state_scratch,
                         size_t scratch_head_dim,
                         int32_t* metadata_validation_scratch,
                         const CpuParallelPtr& cpu_parallel,
                         const float* converted_B,
                         const float* converted_C) {
    OPENVINO_ASSERT(scratch_head_dim > 0 && state_scratch != nullptr);
    OPENVINO_ASSERT(cpu_parallel != nullptr, "PagedSelectiveSSM requires a CPU parallel executor.");
    OPENVINO_ASSERT((converted_B == nullptr) == (converted_C == nullptr));
    const auto metadata = make_paged_metadata(subsequence_begins,
                                              block_indices,
                                              block_indices_begins,
                                              num_processed_tokens,
                                              cache_interval,
                                              index_precision,
                                              shape);
    if (metadata_validation_scratch != nullptr) {
        validate_paged_metadata(metadata, shape, metadata_validation_scratch);
    }
    const PagedSelectiveSSMCallArgs args{A,
                                         dt,
                                         B,
                                         x,
                                         C,
                                         recurrent_state_table,
                                         metadata,
                                         output,
                                         shape,
                                         state_scratch,
                                         scratch_head_dim,
                                         cpu_parallel,
                                         converted_B,
                                         converted_C};
    switch (data_precision) {
    case ov::element::f32:
        dispatch_paged_state_type<float>(args, state_precision);
        return;
    case ov::element::f16:
        dispatch_paged_state_type<ov::float16>(args, state_precision);
        return;
    case ov::element::bf16:
        dispatch_paged_state_type<ov::bfloat16>(args, state_precision);
        return;
    default:
        OPENVINO_THROW("PagedSelectiveSSM supports only f32/f16/bf16 data, got ", data_precision, ".");
    }
}

}  // namespace ov::intel_cpu::node::kernel
