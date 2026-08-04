// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <initializer_list>
#include <vector>

#include "openvino/core/except.hpp"
#include "openvino/core/type/bfloat16.hpp"
#include "openvino/core/type/float16.hpp"
#include "utils/plain_tensor.hpp"

namespace ov::intel_cpu::detail {

inline float read_scalar(const PlainTensor& tensor, const std::initializer_list<size_t>& index) {
    const auto precision = tensor.get_precision();
    if (precision == ov::element::f32) {
        return tensor.at<float>(index);
    }
    if (precision == ov::element::f16) {
        return static_cast<float>(tensor.at<ov::float16>(index));
    }
    if (precision == ov::element::bf16) {
        return static_cast<float>(tensor.at<ov::bfloat16>(index));
    }
    OPENVINO_THROW("Unsupported SelectiveSSM scalar precision: ", precision);
}

inline void write_scalar(const PlainTensor& tensor, const std::initializer_list<size_t>& index, const float value) {
    const auto precision = tensor.get_precision();
    if (precision == ov::element::f32) {
        tensor.at<float>(index) = value;
        return;
    }
    if (precision == ov::element::f16) {
        tensor.at<ov::float16>(index) = static_cast<ov::float16>(value);
        return;
    }
    if (precision == ov::element::bf16) {
        tensor.at<ov::bfloat16>(index) = static_cast<ov::bfloat16>(value);
        return;
    }
    OPENVINO_THROW("Unsupported SelectiveSSM scalar precision: ", precision);
}

inline int64_t read_index_scalar(const PlainTensor& tensor, const size_t idx) {
    const auto precision = tensor.get_precision();
    if (precision == ov::element::i32) {
        return tensor.at<int32_t>({idx});
    }
    if (precision == ov::element::i64) {
        return tensor.at<int64_t>({idx});
    }
    OPENVINO_THROW("Unsupported SelectiveSSM index precision: ", precision);
}

inline void selective_ssm_reference(const PlainTensor& A,
                                    const PlainTensor& dt,
                                    const PlainTensor& B,
                                    const PlainTensor& x,
                                    const PlainTensor& C,
                                    const PlainTensor& recurrent_state,
                                    const PlainTensor& output,
                                    const PlainTensor& output_recurrent_state) {
    const size_t batch = x.size(0);
    const size_t seq_len = x.size(1);
    const size_t num_heads = x.size(2);
    const size_t head_dim = x.size(3);
    const size_t num_groups = B.size(2);
    const size_t state_size = B.size(3);
    const size_t heads_per_group = num_heads / num_groups;

    for (size_t b = 0; b < batch; ++b) {
        for (size_t h = 0; h < num_heads; ++h) {
            for (size_t p = 0; p < head_dim; ++p) {
                for (size_t n = 0; n < state_size; ++n) {
                    write_scalar(output_recurrent_state,
                                 {b, h, p, n},
                                 read_scalar(recurrent_state, {b, h, p, n}));
                }
            }

            const size_t g = h / heads_per_group;
            for (size_t t = 0; t < seq_len; ++t) {
                const float dt_value = read_scalar(dt, {b, t, h});
                const float dA = std::exp(read_scalar(A, {h}) * dt_value);
                for (size_t p = 0; p < head_dim; ++p) {
                    const float x_value = read_scalar(x, {b, t, h, p});
                    float acc = 0.f;
                    for (size_t n = 0; n < state_size; ++n) {
                        const float new_state = read_scalar(output_recurrent_state, {b, h, p, n}) * dA +
                                                x_value * dt_value * read_scalar(B, {b, t, g, n});
                        write_scalar(output_recurrent_state, {b, h, p, n}, new_state);
                        acc += new_state * read_scalar(C, {b, t, g, n});
                    }
                    write_scalar(output, {b, t, h, p}, acc);
                }
            }
        }
    }
}

inline void paged_selective_ssm_reference(const PlainTensor& A,
                                          const PlainTensor& dt,
                                          const PlainTensor& B,
                                          const PlainTensor& x,
                                          const PlainTensor& C,
                                          const PlainTensor& recurrent_state_table,
                                          const PlainTensor& subsequence_begins,
                                          const PlainTensor& block_indices,
                                          const PlainTensor& block_indices_begins,
                                          const PlainTensor& num_processed_tokens,
                                          const PlainTensor& cache_interval,
                                          const PlainTensor& output) {
    const size_t batch_tokens = x.size(0);
    const size_t num_heads = x.size(1);
    const size_t head_dim = x.size(2);
    const size_t num_groups = B.size(1);
    const size_t state_size = B.size(2);
    const size_t heads_per_group = num_heads / num_groups;
    const size_t num_sequences = subsequence_begins.size(0) - 1;

    OPENVINO_ASSERT(batch_tokens == dt.size(0), "PagedSelectiveSSM dt/x token count mismatch");

    for (size_t seq = 0; seq < num_sequences; ++seq) {
        const auto token_begin = static_cast<size_t>(read_index_scalar(subsequence_begins, seq));
        const auto token_end = static_cast<size_t>(read_index_scalar(subsequence_begins, seq + 1));
        const auto block_begin = static_cast<size_t>(read_index_scalar(block_indices_begins, seq));
        const auto block_end = static_cast<size_t>(read_index_scalar(block_indices_begins, seq + 1));
        const auto seq_blocks = block_end > block_begin ? block_end - block_begin : 0;
        const auto processed = static_cast<size_t>(std::max<int64_t>(read_index_scalar(num_processed_tokens, seq), 0));
        const auto interval = static_cast<size_t>(std::max<int64_t>(read_index_scalar(cache_interval, seq), 0));
        const auto prev_nums = interval > 0 ? (processed % interval) : 0;
        const auto first_block = static_cast<size_t>(read_index_scalar(block_indices, block_begin));

        for (size_t h = 0; h < num_heads; ++h) {
            const size_t g = h / heads_per_group;
            std::vector<float> state(head_dim * state_size, 0.f);
            for (size_t p = 0; p < head_dim; ++p) {
                for (size_t n = 0; n < state_size; ++n) {
                    state[p * state_size + n] = read_scalar(recurrent_state_table, {first_block, h, p, n});
                }
            }

            for (size_t token = token_begin; token < token_end; ++token) {
                const float dt_value = read_scalar(dt, {token, h});
                const float dA = std::exp(read_scalar(A, {h}) * dt_value);
                for (size_t p = 0; p < head_dim; ++p) {
                    const float x_value = read_scalar(x, {token, h, p});
                    float acc = 0.f;
                    for (size_t n = 0; n < state_size; ++n) {
                        float& s = state[p * state_size + n];
                        s = s * dA + x_value * dt_value * read_scalar(B, {token, g, n});
                        acc += s * read_scalar(C, {token, g, n});
                    }
                    write_scalar(output, {token, h, p}, acc);
                }

                const size_t processed_tokens = (token - token_begin) + 1;
                const size_t cached_tokens = prev_nums + processed_tokens;
                const bool reached_interval_boundary = interval > 0 && (cached_tokens % interval) == 0;
                const bool reached_sequence_end = token + 1 == token_end;
                if (reached_interval_boundary || reached_sequence_end) {
                    const size_t slot = interval > 0 ? (1 + (cached_tokens - 1) / interval) : 1;
                    if (slot < seq_blocks) {
                        const auto block_id = static_cast<size_t>(read_index_scalar(block_indices, block_begin + slot));
                        for (size_t p = 0; p < head_dim; ++p) {
                            for (size_t n = 0; n < state_size; ++n) {
                                write_scalar(recurrent_state_table, {block_id, h, p, n}, state[p * state_size + n]);
                            }
                        }
                    }
                }
            }
        }
    }
}

}  // namespace ov::intel_cpu::detail
