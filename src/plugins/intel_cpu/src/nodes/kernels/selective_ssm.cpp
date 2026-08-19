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
#include "utils/cpp/bit_cast.hpp"
#include "utils/plain_tensor.hpp"

namespace ov::intel_cpu::node::kernel {
namespace {

// Bound the portable executor's FP32 recurrent-state scratch independently of the CPU cache topology.
// Machine-specific executors may use a cache-aware tiling policy instead.
constexpr size_t max_scratch_bytes_per_worker = 32 * 1024;

template <typename T>
inline float load(const T* ptr) {
    return static_cast<float>(*ptr);
}

// Bulk conversions use cvt_copy below. In the recurrence, conversion is interleaved with arithmetic; keeping it
// inline avoids both an out-of-line ov::float16 scalar call per element and a separate full-state conversion pass.
template <>
inline float load(const ov::float16* ptr) {
    const auto value = bit_cast<uint16_t>(*ptr);
    uint32_t exponent = 0x1FU & (value >> ov::float16::frac_size);
    uint32_t float_exponent = exponent + 127U - ov::float16::exp_bias;
    uint32_t fraction = value & 0x03FFU;
    if (exponent == 0) {
        if (fraction == 0) {
            float_exponent = 0;
        } else {
            ++float_exponent;
            while ((fraction & 0x0400U) == 0) {
                --float_exponent;
                fraction <<= 1U;
            }
            fraction &= 0x03FFU;
        }
    } else if (exponent == 0x1FU) {
        float_exponent = 0xFFU;
    }
    fraction <<= 23U - ov::float16::frac_size;
    const uint32_t bits = (static_cast<uint32_t>(value & 0x8000U) << 16U) | (float_exponent << 23U) | fraction;
    return bit_cast<float>(bits);
}

template <>
inline float load(const ov::bfloat16* ptr) {
    const auto value = bit_cast<uint16_t>(*ptr);
    const uint32_t bits = static_cast<uint32_t>(value) << 16U;
    return bit_cast<float>(bits);
}

template <typename T>
inline void store(T* ptr, float value) {
    *ptr = static_cast<T>(value);
}

template <>
inline void store(ov::float16* ptr, float value) {
    const auto bits = bit_cast<uint32_t>(value);
    constexpr uint32_t sign_mask = 0x80000000U;
    constexpr uint32_t f32_exponent_mask = 0x7F800000U;
    constexpr uint32_t f32_fraction_mask = 0x007FFFFFU;
    constexpr uint32_t f16_exponent_mask = 0x7C000000U;
    constexpr uint32_t f16_fraction_mask = 0x03FF0000U;
    constexpr uint32_t half_round_mask = 0x0001FFFFU;
    constexpr uint32_t normal_round_mask = 0x00007FFFU;
    constexpr uint32_t even_round_bit = 0x00008000U;
    constexpr uint32_t odd_round_value = 0x00018000U;

    const uint32_t biased_exponent_f32 = bits & f32_exponent_mask;
    uint32_t fraction = (bits & f32_fraction_mask) << 3U;
    uint16_t result = 0;
    if (biased_exponent_f32 == f32_exponent_mask) {
        if (fraction != 0) {
            fraction &= f16_fraction_mask;
            if (fraction == 0) {
                fraction = 0x00010000U;
            }
        }
        result = static_cast<uint16_t>(((bits & sign_mask) | f16_exponent_mask | fraction) >> 16U);
    } else if (biased_exponent_f32 == 0) {
        result = static_cast<uint16_t>((bits & sign_mask) >> 16U);
    } else {
        auto biased_exponent_f16 = static_cast<int16_t>(static_cast<int32_t>(biased_exponent_f32 >> 23U) - 127 +
                                                        static_cast<int32_t>(ov::float16::exp_bias));
        if ((fraction & half_round_mask) == odd_round_value || (fraction & normal_round_mask) != 0) {
            fraction += even_round_bit;
            if ((fraction & f16_exponent_mask) != 0) {
                fraction &= f16_exponent_mask;
                ++biased_exponent_f16;
            }
        }
        fraction &= f16_fraction_mask;
        if (biased_exponent_f16 > 30) {
            result = static_cast<uint16_t>(((bits & sign_mask) | f16_exponent_mask) >> 16U);
        } else if (biased_exponent_f16 > 0) {
            result = static_cast<uint16_t>(
                ((bits & sign_mask) | (static_cast<uint32_t>(biased_exponent_f16) << 26U) | fraction) >> 16U);
        } else {
            fraction = 0x04000000U | ((bits & f32_fraction_mask) << 3U);
            const uint32_t shift = biased_exponent_f16 < -30 ? 0U : (uint32_t{1} << (1 - biased_exponent_f16));
            const uint32_t sticky = (fraction & (shift - 1U)) != 0 ? 1U : 0U;
            if (1 + (-biased_exponent_f16) > 31) {
                fraction = 0;
            } else {
                fraction >>= 1 + (-biased_exponent_f16);
            }
            fraction |= sticky;
            if ((fraction & half_round_mask) == odd_round_value || (fraction & normal_round_mask) != 0) {
                fraction += even_round_bit;
            }
            result = static_cast<uint16_t>(((bits & sign_mask) | fraction) >> 16U);
        }
    }
    *ptr = ov::float16::from_bits(result);
}

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

struct PositionReductions {
    float first;
    float second;
};

template <size_t SliceCount, bool StoreState = true, typename StateOutT, typename StateInT, typename ProjectionT>
inline PositionReductions update_state_and_reduce(StateOutT* first_state,
                                                  StateOutT* second_state,
                                                  const StateInT* first_input_state,
                                                  const StateInT* second_input_state,
                                                  const ProjectionT* B,
                                                  const ProjectionT* C,
                                                  float decay,
                                                  float first_input_scale,
                                                  float second_input_scale,
                                                  size_t state_size) {
    static_assert(SliceCount == 1 || SliceCount == 2, "Only the paired path and its single-slice tail are supported");

    // For every contiguous p-slice, compute the recurrence and its C reduction in one state-memory pass:
    //   state[p, n] = state[p, n] * decay + (x[p] * delta) * B[n]
    //   output[p]   = sum_n(state[p, n] * C[n])
    // SliceCount is known at compile time: two slices share each B/C load in the main path, while one slice handles
    // an odd tail through the same implementation. The second pointers and scale are unused for that tail.

    // Four independent sums break the reduction dependency chain. The explicit four-way unroll is a portable C++
    // code-generation aid, not an ISA-specific vector width; specialized executors own explicit SIMD implementations.
    float first_sum0 = 0.F;
    float first_sum1 = 0.F;
    float first_sum2 = 0.F;
    float first_sum3 = 0.F;
    float second_sum0 = 0.F;
    float second_sum1 = 0.F;
    float second_sum2 = 0.F;
    float second_sum3 = 0.F;

    constexpr size_t reduction_unroll = 4;
    size_t n = 0;
    for (; n + reduction_unroll <= state_size; n += reduction_unroll) {
        const float b0 = load(B + n);
        const float b1 = load(B + n + 1);
        const float b2 = load(B + n + 2);
        const float b3 = load(B + n + 3);
        const float c0 = load(C + n);
        const float c1 = load(C + n + 1);
        const float c2 = load(C + n + 2);
        const float c3 = load(C + n + 3);

        const float first_updated0 = load(first_input_state + n) * decay + first_input_scale * b0;
        const float first_updated1 = load(first_input_state + n + 1) * decay + first_input_scale * b1;
        const float first_updated2 = load(first_input_state + n + 2) * decay + first_input_scale * b2;
        const float first_updated3 = load(first_input_state + n + 3) * decay + first_input_scale * b3;
        float second_updated0 = 0.F;
        float second_updated1 = 0.F;
        float second_updated2 = 0.F;
        float second_updated3 = 0.F;
        if constexpr (SliceCount == 2) {
            second_updated0 = load(second_input_state + n) * decay + second_input_scale * b0;
            second_updated1 = load(second_input_state + n + 1) * decay + second_input_scale * b1;
            second_updated2 = load(second_input_state + n + 2) * decay + second_input_scale * b2;
            second_updated3 = load(second_input_state + n + 3) * decay + second_input_scale * b3;
        }

        if constexpr (StoreState) {
            store(first_state + n, first_updated0);
            store(first_state + n + 1, first_updated1);
            store(first_state + n + 2, first_updated2);
            store(first_state + n + 3, first_updated3);
            if constexpr (SliceCount == 2) {
                store(second_state + n, second_updated0);
                store(second_state + n + 1, second_updated1);
                store(second_state + n + 2, second_updated2);
                store(second_state + n + 3, second_updated3);
            }
        }

        first_sum0 += first_updated0 * c0;
        first_sum1 += first_updated1 * c1;
        first_sum2 += first_updated2 * c2;
        first_sum3 += first_updated3 * c3;
        if constexpr (SliceCount == 2) {
            second_sum0 += second_updated0 * c0;
            second_sum1 += second_updated1 * c1;
            second_sum2 += second_updated2 * c2;
            second_sum3 += second_updated3 * c3;
        }
    }

    float first_result = (first_sum0 + first_sum1) + (first_sum2 + first_sum3);
    float second_result = (second_sum0 + second_sum1) + (second_sum2 + second_sum3);
    for (; n < state_size; ++n) {
        const float b_value = load(B + n);
        const float c_value = load(C + n);
        const float first_updated = load(first_input_state + n) * decay + first_input_scale * b_value;
        float second_updated = 0.F;
        if constexpr (SliceCount == 2) {
            second_updated = load(second_input_state + n) * decay + second_input_scale * b_value;
        }

        if constexpr (StoreState) {
            store(first_state + n, first_updated);
            if constexpr (SliceCount == 2) {
                store(second_state + n, second_updated);
            }
        }

        first_result += first_updated * c_value;
        if constexpr (SliceCount == 2) {
            second_result += second_updated * c_value;
        }
    }
    return {first_result, second_result};
}

// Process a block of head-dimension positions as pairs and one optional odd tail. SliceCount is compile-time inside
// update_state_and_reduce, so pairs share B/C loads without adding a runtime branch to the recurrence.
template <bool StoreState = true, typename StateOutT, typename StateInT, typename DataT, typename ProjectionT>
inline void update_state_block_and_reduce(StateOutT* state,
                                          const StateInT* input_state,
                                          const DataT* x,
                                          DataT* output,
                                          const ProjectionT* B,
                                          const ProjectionT* C,
                                          float decay,
                                          float delta,
                                          size_t x_base,
                                          size_t p_count,
                                          size_t state_size) {
    size_t p = 0;
    for (; p + 1 < p_count; p += 2) {
        StateOutT* first_state = nullptr;
        StateOutT* second_state = nullptr;
        if constexpr (StoreState) {
            first_state = state + p * state_size;
            second_state = first_state + state_size;
        }
        const auto* first_input_state = input_state + p * state_size;
        const auto* second_input_state = first_input_state + state_size;
        const float first_input_scale = load(x + x_base + p) * delta;
        const float second_input_scale = load(x + x_base + p + 1) * delta;
        const auto result = update_state_and_reduce<2, StoreState>(first_state,
                                                                   second_state,
                                                                   first_input_state,
                                                                   second_input_state,
                                                                   B,
                                                                   C,
                                                                   decay,
                                                                   first_input_scale,
                                                                   second_input_scale,
                                                                   state_size);
        store(output + x_base + p, result.first);
        store(output + x_base + p + 1, result.second);
    }

    if (p < p_count) {
        StateOutT* tail_state = nullptr;
        if constexpr (StoreState) {
            tail_state = state + p * state_size;
        }
        const float input_scale = load(x + x_base + p) * delta;
        const auto* tail_input_state = input_state + p * state_size;
        const auto result = update_state_and_reduce<1, StoreState>(tail_state,
                                                                   tail_state,
                                                                   tail_input_state,
                                                                   tail_input_state,
                                                                   B,
                                                                   C,
                                                                   decay,
                                                                   input_scale,
                                                                   0.F,
                                                                   state_size);
        store(output + x_base + p, result.first);
    }
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
    // BHS and HS are the recurrent-state strides between consecutive batches and heads, respectively.
    const auto BHS = checked_size_product({shape.num_heads, shape.head_dim, shape.state_size}, "recurrent state batch");
    const auto HS = checked_size_product({shape.head_dim, shape.state_size}, "recurrent state head");
    const auto scratch_stride = checked_size_product({scratch_head_dim, shape.state_size}, "state scratch");
    const auto heads_per_group = shape.num_heads / shape.num_groups;
    const auto p_block_count = ov::util::ceil_div(shape.head_dim, scratch_head_dim);

    cpu_parallel
        ->parallel_for3d(shape.batch_size, shape.num_heads, p_block_count, [&](size_t batch, size_t head, size_t pb) {
            const auto p_begin = pb * scratch_head_dim;
            const auto p_end = p_begin + std::min(scratch_head_dim, shape.head_dim - p_begin);
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

            if (shape.sequence_length == 0) {
                auto* final_state = output_recurrent_state + state_base;
                const auto* initial_state = recurrent_state + state_base;
                if (final_state != initial_state) {
                    std::memcpy(final_state, initial_state, p_count * shape.state_size * sizeof(DataT));
                }
                return;
            }

            const float A_head = load(A + head);
            auto token_head = (batch * shape.sequence_length) * shape.num_heads + head;
            auto projection_base = ((batch * shape.sequence_length) * shape.num_groups + group) * shape.state_size;
            auto x_base = token_head * shape.head_dim + p_begin;
            const auto projection_stride = shape.num_groups * shape.state_size;
            const auto x_stride = shape.num_heads * shape.head_dim;

            if (shape.sequence_length == 1) {
                const float delta = load(dt + token_head);
                const float decay = std::exp(A_head * delta);
                const auto* B_token = B + projection_base;
                const auto* C_token = C + projection_base;
                // Decode has no intermediate FP32 state to preserve. Store the final state directly so the state is
                // converted only once for low precisions and FP32 does not incur an otherwise redundant full copy.
                // Element-wise update remains valid when input and output state alias.
                update_state_block_and_reduce(output_recurrent_state + state_base,
                                              recurrent_state + state_base,
                                              x,
                                              output,
                                              B_token,
                                              C_token,
                                              decay,
                                              delta,
                                              x_base,
                                              p_count,
                                              shape.state_size);
                return;
            }

            // Initialize the working state directly while processing the first token. Apart from removing a full
            // state copy, keeping the first iteration out of the recurrent loop removes the first-token condition
            // from longer sequences.
            {
                const float delta = load(dt + token_head);
                const float decay = std::exp(A_head * delta);
                const auto* B_token = B + projection_base;
                const auto* C_token = C + projection_base;
                update_state_block_and_reduce(local_state,
                                              recurrent_state + state_base,
                                              x,
                                              output,
                                              B_token,
                                              C_token,
                                              decay,
                                              delta,
                                              x_base,
                                              p_count,
                                              shape.state_size);
            }

            token_head += shape.num_heads;
            projection_base += projection_stride;
            x_base += x_stride;
            for (size_t token = 1; token < shape.sequence_length; ++token) {
                const float delta = load(dt + token_head);
                const float decay = std::exp(A_head * delta);
                const auto* B_token = B + projection_base;
                const auto* C_token = C + projection_base;
                update_state_block_and_reduce(local_state,
                                              local_state,
                                              x,
                                              output,
                                              B_token,
                                              C_token,
                                              decay,
                                              delta,
                                              x_base,
                                              p_count,
                                              shape.state_size);
                token_head += shape.num_heads;
                projection_base += projection_stride;
                x_base += x_stride;
            }

            if constexpr (!std::is_same_v<DataT, float>) {
                copy_convert(output_recurrent_state + state_base, local_state, p_count * shape.state_size);
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
    OPENVINO_ASSERT(index_precision == ov::element::i32 || index_precision == ov::element::i64,
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

template <typename DataT, typename ProjectionT>
void paged_selective_ssm_typed(const DataT* A,
                               const DataT* dt,
                               const ProjectionT* B,
                               const DataT* x,
                               const ProjectionT* C,
                               DataT* recurrent_state_table,
                               const PagedMetadata& metadata,
                               DataT* output,
                               const PagedSelectiveSSMShape& shape,
                               float* state_scratch,
                               size_t scratch_head_dim,
                               int32_t* block_owners,
                               const CpuParallelPtr& cpu_parallel) {
    validate_paged_metadata(metadata, shape, block_owners);

    const auto& subsequence_begins = metadata.subsequence_begins;
    const auto& block_indices = metadata.block_indices;
    const auto& block_indices_begins = metadata.block_indices_begins;
    const auto& num_processed_tokens = metadata.num_processed_tokens;
    const auto& cache_interval = metadata.cache_interval;

    const auto block_stride = checked_size_product({shape.num_heads, shape.head_dim, shape.state_size}, "state block");
    const auto head_stride = checked_size_product({shape.head_dim, shape.state_size}, "state head");
    const auto scratch_stride = checked_size_product({scratch_head_dim, shape.state_size}, "state scratch");
    const auto heads_per_group = shape.num_heads / shape.num_groups;
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
            const float A_head = load(A + head);
            const auto interval = index_at(cache_interval, sequence);
            const bool cache_enabled = interval > 0;
            auto token_head = token_begin * shape.num_heads + head;
            auto projection_base = (token_begin * shape.num_groups + group) * shape.state_size;
            auto x_base = token_head * shape.head_dim + p_begin;
            const auto projection_stride = shape.num_groups * shape.state_size;
            const auto x_stride = shape.num_heads * shape.head_dim;

            if (token_end - token_begin == 1) {
                const float delta = load(dt + token_head);
                const float decay = std::exp(A_head * delta);
                const auto* B_token = B + projection_base;
                const auto* C_token = C + projection_base;
                if (cache_enabled) {
                    const auto write_block = static_cast<size_t>(index_at(block_indices, logical_block_begin + 1));
                    auto* state_destination = recurrent_state_table + write_block * block_stride + state_slice;
                    update_state_block_and_reduce(state_destination,
                                                  initial_state,
                                                  x,
                                                  output,
                                                  B_token,
                                                  C_token,
                                                  decay,
                                                  delta,
                                                  x_base,
                                                  p_count,
                                                  shape.state_size);
                } else {
                    // With no cache destination and no following token, the updated state is dead. Compute only
                    // the reduced output and avoid both the scratch write and the final state conversion.
                    update_state_block_and_reduce<false>(static_cast<DataT*>(nullptr),
                                                         initial_state,
                                                         x,
                                                         output,
                                                         B_token,
                                                         C_token,
                                                         decay,
                                                         delta,
                                                         x_base,
                                                         p_count,
                                                         shape.state_size);
                }
                return;
            }

            auto* local_state = state_scratch + static_cast<size_t>(parallel_get_thread_num()) * scratch_stride;
            const uint64_t positive_interval = cache_enabled ? static_cast<uint64_t>(interval) : 1;
            const uint64_t cache_offset =
                cache_enabled ? static_cast<uint64_t>(index_at(num_processed_tokens, sequence)) % positive_interval : 0;

            for (size_t token = token_begin; token < token_end; ++token) {
                const float delta = load(dt + token_head);
                const float decay = std::exp(A_head * delta);
                const auto* B_token = B + projection_base;
                const auto* C_token = C + projection_base;
                const auto update_state = [&](const auto* input_state) {
                    update_state_block_and_reduce(local_state,
                                                  input_state,
                                                  x,
                                                  output,
                                                  B_token,
                                                  C_token,
                                                  decay,
                                                  delta,
                                                  x_base,
                                                  p_count,
                                                  shape.state_size);
                };
                if (token == token_begin) {
                    update_state(initial_state);
                } else {
                    update_state(local_state);
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
                    copy_convert(snapshot, local_state, p_count * shape.state_size);
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
#define OV_CPU_SSM_CALL(ProjectionT, BData, CData)                   \
    selective_ssm_typed(static_cast<const DataT*>(A),                \
                        static_cast<const DataT*>(dt),               \
                        static_cast<const ProjectionT*>(BData),      \
                        static_cast<const DataT*>(x),                \
                        static_cast<const ProjectionT*>(CData),      \
                        static_cast<const DataT*>(recurrent_state),  \
                        static_cast<DataT*>(output),                 \
                        static_cast<DataT*>(output_recurrent_state), \
                        shape,                                       \
                        state_scratch,                               \
                        scratch_head_dim,                            \
                        cpu_parallel)
    if (converted_B != nullptr) {
        OV_CPU_SSM_CALL(float, converted_B, converted_C);
    } else {
        OV_CPU_SSM_CALL(DataT, B, C);
    }
#undef OV_CPU_SSM_CALL
}

template <typename DataT>
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
                               int32_t* block_owners,
                               const CpuParallelPtr& cpu_parallel,
                               const float* converted_B,
                               const float* converted_C) {
#define OV_CPU_PAGED_SSM_CALL(ProjectionT, BData, CData)                  \
    paged_selective_ssm_typed(static_cast<const DataT*>(A),               \
                              static_cast<const DataT*>(dt),              \
                              static_cast<const ProjectionT*>(BData),     \
                              static_cast<const DataT*>(x),               \
                              static_cast<const ProjectionT*>(CData),     \
                              static_cast<DataT*>(recurrent_state_table), \
                              metadata,                                   \
                              static_cast<DataT*>(output),                \
                              shape,                                      \
                              state_scratch,                              \
                              scratch_head_dim,                           \
                              block_owners,                               \
                              cpu_parallel)
    if (converted_B != nullptr) {
        OV_CPU_PAGED_SSM_CALL(float, converted_B, converted_C);
    } else {
        OV_CPU_PAGED_SSM_CALL(DataT, B, C);
    }
#undef OV_CPU_PAGED_SSM_CALL
}

}  // namespace

size_t checked_size_product(std::initializer_list<size_t> dimensions, const char* tensor_name) {
    if (std::find(dimensions.begin(), dimensions.end(), size_t{0}) != dimensions.end()) {
        return 0;
    }

    size_t result = 1;
    for (const auto dimension : dimensions) {
        OPENVINO_ASSERT(result <= std::numeric_limits<size_t>::max() / dimension,
                        "SelectiveSSM size overflow while calculating ",
                        tensor_name,
                        ".");
        result *= dimension;
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

void validate_selective_ssm_shape(const SelectiveSSMShape& shape) {
    OPENVINO_ASSERT(shape.num_groups > 0, "SelectiveSSM num_groups must be greater than zero.");
    OPENVINO_ASSERT(shape.num_heads % shape.num_groups == 0, "SelectiveSSM num_heads must be divisible by num_groups.");
    OPENVINO_ASSERT(shape.state_size > 0, "SelectiveSSM state_size must be greater than zero.");

    checked_size_product({shape.num_heads}, "A");
    checked_size_product({shape.batch_size, shape.sequence_length, shape.num_heads}, "dt");
    checked_size_product({shape.batch_size, shape.sequence_length, shape.num_groups, shape.state_size}, "B/C");
    checked_size_product({shape.batch_size, shape.sequence_length, shape.num_heads, shape.head_dim}, "x/output");
    checked_size_product({shape.batch_size, shape.num_heads, shape.head_dim, shape.state_size}, "recurrent state");
}

void validate_paged_selective_ssm_shape(const PagedSelectiveSSMShape& shape) {
    OPENVINO_ASSERT(shape.num_groups > 0, "PagedSelectiveSSM num_groups must be greater than zero.");
    OPENVINO_ASSERT(shape.num_heads % shape.num_groups == 0,
                    "PagedSelectiveSSM num_heads must be divisible by num_groups.");
    OPENVINO_ASSERT(shape.state_size > 0, "PagedSelectiveSSM state_size must be greater than zero.");

    checked_size_product({shape.num_heads}, "A");
    checked_size_product({shape.token_count, shape.num_heads}, "dt");
    checked_size_product({shape.token_count, shape.num_groups, shape.state_size}, "B/C");
    checked_size_product({shape.token_count, shape.num_heads, shape.head_dim}, "x/output");
    checked_size_product({shape.physical_block_count, shape.num_heads, shape.head_dim, shape.state_size},
                         "recurrent state table");
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
    validate_selective_ssm_shape(shape);
    OPENVINO_ASSERT(scratch_head_dim > 0 && state_scratch != nullptr);
    OPENVINO_ASSERT(cpu_parallel != nullptr, "SelectiveSSM requires a CPU parallel executor.");
    OPENVINO_ASSERT((converted_B == nullptr) == (converted_C == nullptr));
    checked_size_product({scratch_head_dim, shape.state_size}, "state scratch");
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

void validate_paged_selective_ssm_metadata(const void* subsequence_begins,
                                           const void* block_indices,
                                           const void* block_indices_begins,
                                           const void* num_processed_tokens,
                                           const void* cache_interval,
                                           const PagedSelectiveSSMShape& shape,
                                           const ov::element::Type& index_precision,
                                           int32_t* block_owners) {
    validate_paged_metadata(make_paged_metadata(subsequence_begins,
                                                block_indices,
                                                block_indices_begins,
                                                num_processed_tokens,
                                                cache_interval,
                                                index_precision,
                                                shape),
                            shape,
                            block_owners);
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
                         const CpuParallelPtr& cpu_parallel,
                         const float* converted_B,
                         const float* converted_C) {
    validate_paged_selective_ssm_shape(shape);
    OPENVINO_ASSERT(scratch_head_dim > 0 && state_scratch != nullptr);
    OPENVINO_ASSERT(shape.physical_block_count == 0 || block_owners != nullptr);
    OPENVINO_ASSERT(cpu_parallel != nullptr, "PagedSelectiveSSM requires a CPU parallel executor.");
    OPENVINO_ASSERT((converted_B == nullptr) == (converted_C == nullptr));
    checked_size_product({scratch_head_dim, shape.state_size}, "state scratch");
    const auto metadata = make_paged_metadata(subsequence_begins,
                                              block_indices,
                                              block_indices_begins,
                                              num_processed_tokens,
                                              cache_interval,
                                              index_precision,
                                              shape);
#define OV_CPU_PAGED_SSM_CALL(DataT)                        \
    dispatch_paged_projection<DataT>(A,                     \
                                     dt,                    \
                                     B,                     \
                                     x,                     \
                                     C,                     \
                                     recurrent_state_table, \
                                     metadata,              \
                                     output,                \
                                     shape,                 \
                                     state_scratch,         \
                                     scratch_head_dim,      \
                                     block_owners,          \
                                     cpu_parallel,          \
                                     converted_B,           \
                                     converted_C)
    if (precision == ov::element::f32) {
        OV_CPU_PAGED_SSM_CALL(float);
    } else if (precision == ov::element::f16) {
        OV_CPU_PAGED_SSM_CALL(ov::float16);
    } else if (precision == ov::element::bf16) {
        OV_CPU_PAGED_SSM_CALL(ov::bfloat16);
    } else {
        OPENVINO_THROW("PagedSelectiveSSM supports only f32/f16/bf16, got ", precision, ".");
    }
#undef OV_CPU_PAGED_SSM_CALL
}

}  // namespace ov::intel_cpu::node::kernel
