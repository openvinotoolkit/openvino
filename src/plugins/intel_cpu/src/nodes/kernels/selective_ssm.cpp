// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "selective_ssm.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <type_traits>
#include <utility>

#include "openvino/core/except.hpp"
#include "openvino/core/parallel.hpp"
#include "openvino/core/type/bfloat16.hpp"
#include "openvino/core/type/float16.hpp"
#include "utils/cpp/bit_cast.hpp"

namespace ov::intel_cpu::node::kernel {
namespace {

constexpr size_t target_scratch_elements = 8192;

size_t ceil_div(size_t value, size_t divisor) {
    return value / divisor + static_cast<size_t>(value % divisor != 0);
}

template <typename T>
inline float load(const T* ptr) {
    return static_cast<float>(*ptr);
}

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
        int16_t biased_exponent_f16 = static_cast<int16_t>(static_cast<int32_t>(biased_exponent_f32 >> 23U) - 127 +
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

template <typename DataT>
inline void copy_state_to_float(float* dst, const DataT* src, size_t count) {
    if constexpr (std::is_same_v<DataT, float>) {
        if (dst != src) {
            std::memcpy(dst, src, count * sizeof(float));
        }
    } else {
        for (size_t i = 0; i < count; ++i) {
            dst[i] = load(src + i);
        }
    }
}

template <typename DataT>
inline void copy_state_from_float(DataT* dst, const float* src, size_t count) {
    if constexpr (std::is_same_v<DataT, float>) {
        if (dst != src) {
            std::memcpy(dst, src, count * sizeof(float));
        }
    } else {
        for (size_t i = 0; i < count; ++i) {
            store(dst + i, src[i]);
        }
    }
}

template <bool StoreState = true, typename StateOutT, typename StateInT, typename ProjectionT>
inline float update_state_and_reduce(StateOutT* state,
                                     const StateInT* input_state,
                                     const ProjectionT* B,
                                     const ProjectionT* C,
                                     float decay,
                                     float input_scale,
                                     size_t state_size) {
    // Four independent reduction chains expose instruction-level parallelism to any optimizing compiler. Supporting
    // different input/output types also lets decode write its final state directly without an intermediate copy.
    float result0 = 0.F;
    float result1 = 0.F;
    float result2 = 0.F;
    float result3 = 0.F;
    size_t n = 0;
    for (; n + 4 <= state_size; n += 4) {
        const float updated_state0 = load(input_state + n) * decay + input_scale * load(B + n);
        const float updated_state1 = load(input_state + n + 1) * decay + input_scale * load(B + n + 1);
        const float updated_state2 = load(input_state + n + 2) * decay + input_scale * load(B + n + 2);
        const float updated_state3 = load(input_state + n + 3) * decay + input_scale * load(B + n + 3);
        if constexpr (StoreState) {
            store(state + n, updated_state0);
            store(state + n + 1, updated_state1);
            store(state + n + 2, updated_state2);
            store(state + n + 3, updated_state3);
        }
        result0 += updated_state0 * load(C + n);
        result1 += updated_state1 * load(C + n + 1);
        result2 += updated_state2 * load(C + n + 2);
        result3 += updated_state3 * load(C + n + 3);
    }
    float result = (result0 + result1) + (result2 + result3);
    for (; n < state_size; ++n) {
        const float updated_state = load(input_state + n) * decay + input_scale * load(B + n);
        if constexpr (StoreState) {
            store(state + n, updated_state);
        }
        result += updated_state * load(C + n);
    }
    return result;
}

template <bool StoreState = true, typename StateOutT, typename StateInT, typename ProjectionT>
inline std::pair<float, float> update_state_pair_and_reduce(StateOutT* state0,
                                                            StateOutT* state1,
                                                            const StateInT* input_state0,
                                                            const StateInT* input_state1,
                                                            const ProjectionT* B,
                                                            const ProjectionT* C,
                                                            float decay,
                                                            float input_scale0,
                                                            float input_scale1,
                                                            size_t state_size) {
    float result00 = 0.F;
    float result01 = 0.F;
    float result02 = 0.F;
    float result03 = 0.F;
    float result10 = 0.F;
    float result11 = 0.F;
    float result12 = 0.F;
    float result13 = 0.F;
    size_t n = 0;
    for (; n + 4 <= state_size; n += 4) {
        const float b0 = load(B + n);
        const float b1 = load(B + n + 1);
        const float b2 = load(B + n + 2);
        const float b3 = load(B + n + 3);
        const float c0 = load(C + n);
        const float c1 = load(C + n + 1);
        const float c2 = load(C + n + 2);
        const float c3 = load(C + n + 3);
        const float updated00 = load(input_state0 + n) * decay + input_scale0 * b0;
        const float updated01 = load(input_state0 + n + 1) * decay + input_scale0 * b1;
        const float updated02 = load(input_state0 + n + 2) * decay + input_scale0 * b2;
        const float updated03 = load(input_state0 + n + 3) * decay + input_scale0 * b3;
        const float updated10 = load(input_state1 + n) * decay + input_scale1 * b0;
        const float updated11 = load(input_state1 + n + 1) * decay + input_scale1 * b1;
        const float updated12 = load(input_state1 + n + 2) * decay + input_scale1 * b2;
        const float updated13 = load(input_state1 + n + 3) * decay + input_scale1 * b3;
        if constexpr (StoreState) {
            store(state0 + n, updated00);
            store(state0 + n + 1, updated01);
            store(state0 + n + 2, updated02);
            store(state0 + n + 3, updated03);
            store(state1 + n, updated10);
            store(state1 + n + 1, updated11);
            store(state1 + n + 2, updated12);
            store(state1 + n + 3, updated13);
        }
        result00 += updated00 * c0;
        result01 += updated01 * c1;
        result02 += updated02 * c2;
        result03 += updated03 * c3;
        result10 += updated10 * c0;
        result11 += updated11 * c1;
        result12 += updated12 * c2;
        result13 += updated13 * c3;
    }
    float result0 = (result00 + result01) + (result02 + result03);
    float result1 = (result10 + result11) + (result12 + result13);
    for (; n < state_size; ++n) {
        const float b = load(B + n);
        const float c = load(C + n);
        const float updated0 = load(input_state0 + n) * decay + input_scale0 * b;
        const float updated1 = load(input_state1 + n) * decay + input_scale1 * b;
        if constexpr (StoreState) {
            store(state0 + n, updated0);
            store(state1 + n, updated1);
        }
        result0 += updated0 * c;
        result1 += updated1 * c;
    }
    return {result0, result1};
}

template <bool StoreState = true, typename StateOutT, typename StateInT, typename DataT, typename ProjectionT>
inline void update_state_slices_and_reduce(StateOutT* state,
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
        StateOutT* state0 = nullptr;
        StateOutT* state1 = nullptr;
        if constexpr (StoreState) {
            state0 = state + p * state_size;
            state1 = state0 + state_size;
        }
        const auto* input_state0 = input_state + p * state_size;
        const auto* input_state1 = input_state0 + state_size;
        const float input_scale0 = load(x + x_base + p) * delta;
        const float input_scale1 = load(x + x_base + p + 1) * delta;
        const auto result = update_state_pair_and_reduce<StoreState>(state0,
                                                                     state1,
                                                                     input_state0,
                                                                     input_state1,
                                                                     B,
                                                                     C,
                                                                     decay,
                                                                     input_scale0,
                                                                     input_scale1,
                                                                     state_size);
        store(output + x_base + p, result.first);
        store(output + x_base + p + 1, result.second);
    }
    if (p < p_count) {
        const float input_scale = load(x + x_base + p) * delta;
        StateOutT* state_tail = nullptr;
        if constexpr (StoreState) {
            state_tail = state + p * state_size;
        }
        const float result = update_state_and_reduce<StoreState>(state_tail,
                                                                 input_state + p * state_size,
                                                                 B,
                                                                 C,
                                                                 decay,
                                                                 input_scale,
                                                                 state_size);
        store(output + x_base + p, result);
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
    const auto BHS = checked_size_product({shape.num_heads, shape.head_dim, shape.state_size}, "recurrent state batch");
    const auto HS = checked_size_product({shape.head_dim, shape.state_size}, "recurrent state head");
    const auto scratch_stride = checked_size_product({scratch_head_dim, shape.state_size}, "state scratch");
    const auto heads_per_group = shape.num_heads / shape.num_groups;
    const auto p_block_count = ceil_div(shape.head_dim, scratch_head_dim);

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
                if constexpr (std::is_same_v<DataT, float>) {
                    copy_state_to_float(local_state, recurrent_state + state_base, p_count * shape.state_size);
                    update_state_slices_and_reduce(local_state,
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
                } else {
                    // Unlike prefill, decode does not need persistent FP32 state between tokens. Write the final
                    // low-precision state directly and avoid conversion passes through the per-worker scratchpad.
                    update_state_slices_and_reduce(output_recurrent_state + state_base,
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
                update_state_slices_and_reduce(local_state,
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
                update_state_slices_and_reduce(local_state,
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
                copy_state_from_float(output_recurrent_state + state_base, local_state, p_count * shape.state_size);
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

template <typename DataT, typename ProjectionT, typename IndexT>
void paged_selective_ssm_typed(const DataT* A,
                               const DataT* dt,
                               const ProjectionT* B,
                               const DataT* x,
                               const ProjectionT* C,
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

    const auto block_stride = checked_size_product({shape.num_heads, shape.head_dim, shape.state_size}, "state block");
    const auto head_stride = checked_size_product({shape.head_dim, shape.state_size}, "state head");
    const auto scratch_stride = checked_size_product({scratch_head_dim, shape.state_size}, "state scratch");
    const auto heads_per_group = shape.num_heads / shape.num_groups;
    const auto p_block_count = ceil_div(shape.head_dim, scratch_head_dim);

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
            const float A_head = load(A + head);
            const auto interval = cache_interval[sequence];
            const bool cache_enabled = interval > 0;
            auto token_head = token_begin * shape.num_heads + head;
            auto projection_base = (token_begin * shape.num_groups + group) * shape.state_size;
            auto x_base = token_head * shape.head_dim + p_begin;
            const auto projection_stride = shape.num_groups * shape.state_size;
            const auto x_stride = shape.num_heads * shape.head_dim;

            const auto sequence_length = token_end - token_begin;
            if (sequence_length == 1) {
                const float delta = load(dt + token_head);
                const float decay = std::exp(A_head * delta);
                const auto* B_token = B + projection_base;
                const auto* C_token = C + projection_base;
                if (cache_enabled) {
                    const auto write_block = static_cast<size_t>(block_indices[logical_block_begin + 1]);
                    auto* state_destination = recurrent_state_table + write_block * block_stride + state_slice;
                    update_state_slices_and_reduce(state_destination,
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
                    // With no cache destination and no following token, the updated state is dead. Reuse the exact
                    // same recurrence with stores disabled at compile time and produce only the reduced output.
                    update_state_slices_and_reduce<false>(static_cast<float*>(nullptr),
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
                cache_enabled ? static_cast<uint64_t>(num_processed_tokens[sequence]) % positive_interval : 0;
            // Track cache boundaries incrementally to keep integer division out of the token loop.
            uint64_t tokens_until_boundary = cache_enabled ? positive_interval - cache_offset : 0;
            size_t write_slot = 1;

            // Form the FP32 working state directly from the initial state and the first token. This removes the
            // complete initial-state copy and keeps the remaining token loop type-homogeneous.
            {
                const float delta = load(dt + token_head);
                const float decay = std::exp(A_head * delta);
                const auto* B_token = B + projection_base;
                const auto* C_token = C + projection_base;
                update_state_slices_and_reduce(local_state,
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

                if (cache_enabled && --tokens_until_boundary == 0) {
                    const auto write_block = static_cast<size_t>(block_indices[logical_block_begin + write_slot++]);
                    auto* snapshot = recurrent_state_table + write_block * block_stride + state_slice;
                    copy_state_from_float(snapshot, local_state, p_count * shape.state_size);
                    tokens_until_boundary = positive_interval;
                }
            }

            token_head += shape.num_heads;
            projection_base += projection_stride;
            x_base += x_stride;
            for (size_t token = token_begin + 1; token < token_end; ++token) {
                const float delta = load(dt + token_head);
                const float decay = std::exp(A_head * delta);
                const auto* B_token = B + projection_base;
                const auto* C_token = C + projection_base;
                update_state_slices_and_reduce(local_state,
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

template <typename DataT, typename ProjectionT>
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
                              static_cast<const ProjectionT*>(B),               \
                              static_cast<const DataT*>(x),                     \
                              static_cast<const ProjectionT*>(C),               \
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

template <typename DataT>
void dispatch_paged_projection(const void* A,
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
                               const CpuParallelPtr& cpu_parallel,
                               const float* converted_B,
                               const float* converted_C) {
#define OV_CPU_PAGED_SSM_CALL(ProjectionT, BData, CData)            \
    dispatch_paged_index<DataT, ProjectionT>(A,                     \
                                             dt,                    \
                                             BData,                 \
                                             x,                     \
                                             CData,                 \
                                             recurrent_state_table, \
                                             subsequence_begins,    \
                                             block_indices,         \
                                             block_indices_begins,  \
                                             num_processed_tokens,  \
                                             cache_interval,        \
                                             output,                \
                                             shape,                 \
                                             index_precision,       \
                                             state_scratch,         \
                                             scratch_head_dim,      \
                                             block_owners,          \
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
    const auto cache_limited = std::max(size_t{1}, std::min(head_dim, target_scratch_elements / state_size));
    const auto outer_work = std::max(size_t{1}, outer_work_items);
    const auto workers = std::max(size_t{1}, thread_count);
    const auto blocks_for_parallelism = ceil_div(workers, outer_work);
    const auto parallelism_limited = ceil_div(head_dim, blocks_for_parallelism);
    return std::max(size_t{1}, std::min(cache_limited, parallelism_limited));
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
    if (index_precision == ov::element::i32) {
        validate_paged_metadata(static_cast<const int32_t*>(subsequence_begins),
                                static_cast<const int32_t*>(block_indices),
                                static_cast<const int32_t*>(block_indices_begins),
                                static_cast<const int32_t*>(num_processed_tokens),
                                static_cast<const int32_t*>(cache_interval),
                                shape,
                                block_owners);
    } else if (index_precision == ov::element::i64) {
        validate_paged_metadata(static_cast<const int64_t*>(subsequence_begins),
                                static_cast<const int64_t*>(block_indices),
                                static_cast<const int64_t*>(block_indices_begins),
                                static_cast<const int64_t*>(num_processed_tokens),
                                static_cast<const int64_t*>(cache_interval),
                                shape,
                                block_owners);
    } else {
        OPENVINO_THROW("PagedSelectiveSSM supports only i32/i64 metadata, got ", index_precision, ".");
    }
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
    OPENVINO_ASSERT((converted_B == nullptr) == (converted_C == nullptr));
    checked_size_product({scratch_head_dim, shape.state_size}, "state scratch");
#define OV_CPU_PAGED_SSM_CALL(DataT)                        \
    dispatch_paged_projection<DataT>(A,                     \
                                     dt,                    \
                                     B,                     \
                                     x,                     \
                                     C,                     \
                                     recurrent_state_table, \
                                     subsequence_begins,    \
                                     block_indices,         \
                                     block_indices_begins,  \
                                     num_processed_tokens,  \
                                     cache_interval,        \
                                     output,                \
                                     shape,                 \
                                     index_precision,       \
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
