// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "selective_ssm.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>

#include "openvino/core/except.hpp"
#include "openvino/core/parallel.hpp"
#include "openvino/core/type/bfloat16.hpp"
#include "openvino/core/type/float16.hpp"

namespace ov::intel_cpu::node::kernel {
namespace {

template <typename T>
inline float load(const T* ptr) {
    return static_cast<float>(*ptr);
}

template <typename T>
inline void store(T* ptr, float value) {
    *ptr = static_cast<T>(value);
}

template <typename DataT>
inline void copy_state_to_float(float* dst, const DataT* src, size_t count) {
    if constexpr (std::is_same_v<DataT, float>) {
        if (dst == src) {
            return;
        }
    }
    for (size_t i = 0; i < count; ++i) {
        dst[i] = load(src + i);
    }
}

template <typename DataT>
inline float update_state_and_reduce(float* state,
                                     const DataT* B,
                                     const DataT* C,
                                     float decay,
                                     float input_scale,
                                     size_t state_size) {
    // Keep four independent reduction chains to expose instruction-level parallelism to any optimizing compiler
    // without relying on a target ISA. Fusing the update and reduction also keeps the new state cache-resident.
    float result0 = 0.F;
    float result1 = 0.F;
    float result2 = 0.F;
    float result3 = 0.F;
    size_t n = 0;
    for (; n + 4 <= state_size; n += 4) {
        const float updated_state0 = state[n] * decay + input_scale * load(B + n);
        const float updated_state1 = state[n + 1] * decay + input_scale * load(B + n + 1);
        const float updated_state2 = state[n + 2] * decay + input_scale * load(B + n + 2);
        const float updated_state3 = state[n + 3] * decay + input_scale * load(B + n + 3);
        state[n] = updated_state0;
        state[n + 1] = updated_state1;
        state[n + 2] = updated_state2;
        state[n + 3] = updated_state3;
        result0 += updated_state0 * load(C + n);
        result1 += updated_state1 * load(C + n + 1);
        result2 += updated_state2 * load(C + n + 2);
        result3 += updated_state3 * load(C + n + 3);
    }
    float result = (result0 + result1) + (result2 + result3);
    for (; n < state_size; ++n) {
        const float updated_state = state[n] * decay + input_scale * load(B + n);
        state[n] = updated_state;
        result += updated_state * load(C + n);
    }
    return result;
}

template <typename DataT>
void selective_ssm_typed(const DataT* A,
                         const DataT* dt,
                         const DataT* B,
                         const DataT* x,
                         const DataT* C,
                         const DataT* recurrent_state,
                         DataT* output,
                         DataT* output_recurrent_state,
                         const SelectiveSSMShape& shape,
                         float* state_scratch,
                         size_t scratch_head_dim,
                         const CpuParallelPtr& cpu_parallel) {
    const auto BHS = shape.num_heads * shape.head_dim * shape.state_size;
    const auto HS = shape.head_dim * shape.state_size;
    const auto scratch_stride = scratch_head_dim * shape.state_size;
    const auto heads_per_group = shape.num_heads / shape.num_groups;
    const auto p_block_count = (shape.head_dim + scratch_head_dim - 1) / scratch_head_dim;

    cpu_parallel
        ->parallel_for3d(shape.batch_size, shape.num_heads, p_block_count, [&](size_t batch, size_t head, size_t pb) {
            const auto p_begin = pb * scratch_head_dim;
            const auto p_end = std::min(p_begin + scratch_head_dim, shape.head_dim);
            const auto p_count = p_end - p_begin;
            const auto group = head / heads_per_group;
            const auto state_base = batch * BHS + head * HS + p_begin * shape.state_size;
            float* local_state = nullptr;
            if constexpr (std::is_same_v<DataT, float>) {
                // FP32 can use the final state as its working buffer. Lower precisions retain FP32 scratch to avoid
                // quantizing the recurrent state at every token.
                local_state = output_recurrent_state + state_base;
            } else {
                local_state = state_scratch + static_cast<size_t>(parallel_get_thread_num()) * scratch_stride;
            }

            for (size_t p = 0; p < p_count; ++p) {
                const auto* src = recurrent_state + state_base + p * shape.state_size;
                auto* dst = local_state + p * shape.state_size;
                copy_state_to_float(dst, src, shape.state_size);
            }

            const float A_head = load(A + head);
            auto token_head = (batch * shape.sequence_length) * shape.num_heads + head;
            auto projection_base = ((batch * shape.sequence_length) * shape.num_groups + group) * shape.state_size;
            auto x_base = token_head * shape.head_dim + p_begin;
            const auto projection_stride = shape.num_groups * shape.state_size;
            const auto x_stride = shape.num_heads * shape.head_dim;
            for (size_t token = 0; token < shape.sequence_length; ++token) {
                const float delta = load(dt + token_head);
                const float decay = std::exp(A_head * delta);
                const auto* B_token = B + projection_base;
                const auto* C_token = C + projection_base;

                for (size_t p = 0; p < p_count; ++p) {
                    auto* state = local_state + p * shape.state_size;
                    const float input_scale = load(x + x_base + p) * delta;
                    const float result =
                        update_state_and_reduce(state, B_token, C_token, decay, input_scale, shape.state_size);
                    store(output + x_base + p, result);
                }
                token_head += shape.num_heads;
                projection_base += projection_stride;
                x_base += x_stride;
            }

            if constexpr (!std::is_same_v<DataT, float>) {
                for (size_t p = 0; p < p_count; ++p) {
                    const auto* src = local_state + p * shape.state_size;
                    auto* dst = output_recurrent_state + state_base + p * shape.state_size;
                    for (size_t n = 0; n < shape.state_size; ++n) {
                        store(dst + n, src[n]);
                    }
                }
            }
        });
}

template <typename IndexT>
size_t checked_index(IndexT value, size_t limit, const char* name, size_t position) {
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

template <typename IndexT>
void validate_paged_metadata(const IndexT* subsequence_begins,
                             const IndexT* block_indices,
                             const IndexT* block_indices_begins,
                             const IndexT* num_processed_tokens,
                             const IndexT* cache_interval,
                             const PagedSelectiveSSMShape& shape,
                             int32_t* block_owners) {
    OPENVINO_ASSERT(subsequence_begins[0] == 0,
                    "PagedSelectiveSSM: subsequence_begins[0] must be 0, got ",
                    subsequence_begins[0],
                    ".");
    OPENVINO_ASSERT(block_indices_begins[0] == 0,
                    "PagedSelectiveSSM: block_indices_begins[0] must be 0, got ",
                    block_indices_begins[0],
                    ".");

    for (size_t s = 0; s < shape.sequence_count; ++s) {
        OPENVINO_ASSERT(subsequence_begins[s] >= 0 && subsequence_begins[s + 1] >= subsequence_begins[s],
                        "PagedSelectiveSSM: subsequence_begins must be non-negative and non-decreasing at sequence ",
                        s,
                        ".");
        OPENVINO_ASSERT(block_indices_begins[s] >= 0 && block_indices_begins[s + 1] >= block_indices_begins[s],
                        "PagedSelectiveSSM: block_indices_begins must be non-negative and non-decreasing at sequence ",
                        s,
                        ".");
        OPENVINO_ASSERT(num_processed_tokens[s] >= 0,
                        "PagedSelectiveSSM: num_processed_tokens[",
                        s,
                        "] must be non-negative, got ",
                        num_processed_tokens[s],
                        ".");
    }

    OPENVINO_ASSERT(static_cast<uint64_t>(subsequence_begins[shape.sequence_count]) == shape.token_count,
                    "PagedSelectiveSSM: the last subsequence offset must equal token_count (",
                    shape.token_count,
                    "), got ",
                    subsequence_begins[shape.sequence_count],
                    ".");
    OPENVINO_ASSERT(static_cast<uint64_t>(block_indices_begins[shape.sequence_count]) == shape.logical_block_count,
                    "PagedSelectiveSSM: the last block offset must equal logical_block_count (",
                    shape.logical_block_count,
                    "), got ",
                    block_indices_begins[shape.sequence_count],
                    ".");

    for (size_t i = 0; i < shape.logical_block_count; ++i) {
        checked_index(block_indices[i], shape.physical_block_count, "block_indices", i);
    }
    OPENVINO_ASSERT(shape.sequence_count <= static_cast<size_t>(std::numeric_limits<int32_t>::max()),
                    "PagedSelectiveSSM supports at most INT32_MAX sequences.");
    if (shape.physical_block_count > 0) {
        std::fill(block_owners, block_owners + shape.physical_block_count, int32_t{-1});
    }

    // First register all physical blocks that will be written. Two different sequences
    // must never write the same block, even when their head/head-dim slices are disjoint tasks.
    for (size_t s = 0; s < shape.sequence_count; ++s) {
        const auto token_begin = static_cast<uint64_t>(subsequence_begins[s]);
        const auto token_end = static_cast<uint64_t>(subsequence_begins[s + 1]);
        const auto token_count = token_end - token_begin;
        if (token_count == 0) {
            continue;
        }

        const auto block_begin = static_cast<uint64_t>(block_indices_begins[s]);
        const auto block_end = static_cast<uint64_t>(block_indices_begins[s + 1]);
        OPENVINO_ASSERT(block_end > block_begin,
                        "PagedSelectiveSSM: non-empty sequence ",
                        s,
                        " requires a read block.");

        const auto interval = cache_interval[s];
        if (interval <= 0) {
            continue;
        }
        const auto positive_interval = static_cast<uint64_t>(interval);
        const auto offset = static_cast<uint64_t>(num_processed_tokens[s]) % positive_interval;
        OPENVINO_ASSERT(token_count <= std::numeric_limits<uint64_t>::max() - offset,
                        "PagedSelectiveSSM: token count overflow at sequence ",
                        s,
                        ".");
        const auto cached_token_count = offset + token_count;
        const auto write_count = (cached_token_count - 1) / positive_interval + 1;
        OPENVINO_ASSERT(block_end - block_begin >= write_count + 1,
                        "PagedSelectiveSSM: sequence ",
                        s,
                        " requires ",
                        write_count + 1,
                        " logical blocks (one read plus ",
                        write_count,
                        " writes), got ",
                        block_end - block_begin,
                        ".");

        for (uint64_t slot = 1; slot <= write_count; ++slot) {
            const auto logical = static_cast<size_t>(block_begin + slot);
            const auto physical =
                checked_index(block_indices[logical], shape.physical_block_count, "block_indices", logical);
            OPENVINO_ASSERT(block_owners[physical] == -1 || block_owners[physical] == static_cast<int32_t>(s),
                            "PagedSelectiveSSM: physical block ",
                            physical,
                            " is written by multiple sequences (",
                            block_owners[physical],
                            " and ",
                            s,
                            ").");
            block_owners[physical] = static_cast<int32_t>(s);
        }
    }

    // A read may alias this sequence's first write (the documented in-place case), but it
    // must not race with another sequence's writer. Read/read sharing is harmless.
    for (size_t s = 0; s < shape.sequence_count; ++s) {
        if (subsequence_begins[s + 1] == subsequence_begins[s]) {
            continue;
        }
        const auto logical = static_cast<size_t>(block_indices_begins[s]);
        const auto physical =
            checked_index(block_indices[logical], shape.physical_block_count, "block_indices", logical);
        OPENVINO_ASSERT(block_owners[physical] == -1 || block_owners[physical] == static_cast<int32_t>(s),
                        "PagedSelectiveSSM: sequence ",
                        s,
                        " reads physical block ",
                        physical,
                        " while sequence ",
                        block_owners[physical],
                        " writes it.");
    }
}

template <typename DataT, typename IndexT>
void paged_selective_ssm_typed(const DataT* A,
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
                               const PagedSelectiveSSMShape& shape,
                               float* state_scratch,
                               size_t scratch_head_dim,
                               int32_t* block_owners,
                               const CpuParallelPtr& cpu_parallel) {
    validate_paged_metadata(subsequence_begins,
                            block_indices,
                            block_indices_begins,
                            num_processed_tokens,
                            cache_interval,
                            shape,
                            block_owners);

    const auto block_stride = shape.num_heads * shape.head_dim * shape.state_size;
    const auto head_stride = shape.head_dim * shape.state_size;
    const auto scratch_stride = scratch_head_dim * shape.state_size;
    const auto heads_per_group = shape.num_heads / shape.num_groups;
    const auto p_block_count = (shape.head_dim + scratch_head_dim - 1) / scratch_head_dim;

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
            const auto p_end = std::min(p_begin + scratch_head_dim, shape.head_dim);
            const auto p_count = p_end - p_begin;
            const auto group = head / heads_per_group;
            const auto logical_block_begin = static_cast<size_t>(block_indices_begins[sequence]);
            const auto read_block = static_cast<size_t>(block_indices[logical_block_begin]);
            auto* local_state = state_scratch + static_cast<size_t>(parallel_get_thread_num()) * scratch_stride;
            const auto state_slice = head * head_stride + p_begin * shape.state_size;
            const auto* initial_state = recurrent_state_table + read_block * block_stride + state_slice;

            for (size_t p = 0; p < p_count; ++p) {
                const auto* src = initial_state + p * shape.state_size;
                auto* dst = local_state + p * shape.state_size;
                for (size_t n = 0; n < shape.state_size; ++n) {
                    dst[n] = load(src + n);
                }
            }

            const float A_head = load(A + head);
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
            // Track cache boundaries incrementally to keep integer division out of the token loop.
            uint64_t tokens_until_boundary = cache_enabled ? positive_interval - cache_offset : 0;
            size_t write_slot = 1;
            for (size_t token = token_begin; token < token_end; ++token) {
                const float delta = load(dt + token_head);
                const float decay = std::exp(A_head * delta);
                const auto* B_token = B + projection_base;
                const auto* C_token = C + projection_base;

                for (size_t p = 0; p < p_count; ++p) {
                    auto* state = local_state + p * shape.state_size;
                    const float input_scale = load(x + x_base + p) * delta;
                    const float result =
                        update_state_and_reduce(state, B_token, C_token, decay, input_scale, shape.state_size);
                    store(output + x_base + p, result);
                }

                if (cache_enabled) {
                    const bool is_boundary = --tokens_until_boundary == 0;
                    const bool is_last = token + 1 == token_end;
                    if (is_boundary || is_last) {
                        const auto write_block = static_cast<size_t>(block_indices[logical_block_begin + write_slot++]);
                        auto* snapshot = recurrent_state_table + write_block * block_stride + state_slice;
                        for (size_t p = 0; p < p_count; ++p) {
                            const auto* src = local_state + p * shape.state_size;
                            auto* dst = snapshot + p * shape.state_size;
                            for (size_t n = 0; n < shape.state_size; ++n) {
                                store(dst + n, src[n]);
                            }
                        }
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

template <typename DataT>
void dispatch_paged_index(const void* A,
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
                          const ov::element::Type& index_precision,
                          float* state_scratch,
                          size_t scratch_head_dim,
                          int32_t* block_owners,
                          const CpuParallelPtr& cpu_parallel) {
#define OV_CPU_PAGED_SSM_CALL(IndexT)                                           \
    paged_selective_ssm_typed(static_cast<const DataT*>(A),                     \
                              static_cast<const DataT*>(dt),                    \
                              static_cast<const DataT*>(B),                     \
                              static_cast<const DataT*>(x),                     \
                              static_cast<const DataT*>(C),                     \
                              static_cast<DataT*>(recurrent_state_table),       \
                              static_cast<const IndexT*>(subsequence_begins),   \
                              static_cast<const IndexT*>(block_indices),        \
                              static_cast<const IndexT*>(block_indices_begins), \
                              static_cast<const IndexT*>(num_processed_tokens), \
                              static_cast<const IndexT*>(cache_interval),       \
                              static_cast<DataT*>(output),                      \
                              shape,                                            \
                              state_scratch,                                    \
                              scratch_head_dim,                                 \
                              block_owners,                                     \
                              cpu_parallel)
    if (index_precision == ov::element::i32) {
        OV_CPU_PAGED_SSM_CALL(int32_t);
    } else if (index_precision == ov::element::i64) {
        OV_CPU_PAGED_SSM_CALL(int64_t);
    } else {
        OPENVINO_THROW("PagedSelectiveSSM supports only i32/i64 metadata, got ", index_precision, ".");
    }
#undef OV_CPU_PAGED_SSM_CALL
}

}  // namespace

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
                   const CpuParallelPtr& cpu_parallel) {
    OPENVINO_ASSERT(shape.num_groups > 0 && shape.num_heads % shape.num_groups == 0);
    OPENVINO_ASSERT(shape.state_size > 0 && scratch_head_dim > 0 && state_scratch != nullptr);
#define OV_CPU_SSM_CALL(DataT)                                       \
    selective_ssm_typed(static_cast<const DataT*>(A),                \
                        static_cast<const DataT*>(dt),               \
                        static_cast<const DataT*>(B),                \
                        static_cast<const DataT*>(x),                \
                        static_cast<const DataT*>(C),                \
                        static_cast<const DataT*>(recurrent_state),  \
                        static_cast<DataT*>(output),                 \
                        static_cast<DataT*>(output_recurrent_state), \
                        shape,                                       \
                        state_scratch,                               \
                        scratch_head_dim,                            \
                        cpu_parallel)
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
                         const ov::element::Type& precision,
                         const ov::element::Type& index_precision,
                         float* state_scratch,
                         size_t scratch_head_dim,
                         int32_t* block_owners,
                         const CpuParallelPtr& cpu_parallel) {
    OPENVINO_ASSERT(shape.num_groups > 0 && shape.num_heads % shape.num_groups == 0);
    OPENVINO_ASSERT(shape.state_size > 0 && scratch_head_dim > 0 && state_scratch != nullptr);
    OPENVINO_ASSERT(shape.physical_block_count == 0 || block_owners != nullptr);
    if (precision == ov::element::f32) {
        dispatch_paged_index<float>(A,
                                    dt,
                                    B,
                                    x,
                                    C,
                                    recurrent_state_table,
                                    subsequence_begins,
                                    block_indices,
                                    block_indices_begins,
                                    num_processed_tokens,
                                    cache_interval,
                                    output,
                                    shape,
                                    index_precision,
                                    state_scratch,
                                    scratch_head_dim,
                                    block_owners,
                                    cpu_parallel);
    } else if (precision == ov::element::f16) {
        dispatch_paged_index<ov::float16>(A,
                                          dt,
                                          B,
                                          x,
                                          C,
                                          recurrent_state_table,
                                          subsequence_begins,
                                          block_indices,
                                          block_indices_begins,
                                          num_processed_tokens,
                                          cache_interval,
                                          output,
                                          shape,
                                          index_precision,
                                          state_scratch,
                                          scratch_head_dim,
                                          block_owners,
                                          cpu_parallel);
    } else if (precision == ov::element::bf16) {
        dispatch_paged_index<ov::bfloat16>(A,
                                           dt,
                                           B,
                                           x,
                                           C,
                                           recurrent_state_table,
                                           subsequence_begins,
                                           block_indices,
                                           block_indices_begins,
                                           num_processed_tokens,
                                           cache_interval,
                                           output,
                                           shape,
                                           index_precision,
                                           state_scratch,
                                           scratch_head_dim,
                                           block_owners,
                                           cpu_parallel);
    } else {
        OPENVINO_THROW("PagedSelectiveSSM supports only f32/f16/bf16, got ", precision, ".");
    }
}

}  // namespace ov::intel_cpu::node::kernel
