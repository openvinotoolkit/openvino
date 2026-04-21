// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

#include "attn_quant_kernel.hpp"
#include "common.hpp"
#include "cpu_parallel.hpp"
#include "openvino/core/except.hpp"
#include "utils/plain_tensor.hpp"

namespace ov::Extensions::Cpu::XARCH {

using ov::intel_cpu::CpuParallelPtr;
using ov::intel_cpu::PlainTensor;

// Helper: dispatch runtime precision to compile-time template
template <typename Func>
inline void dispatch_precision(ov::element::Type_t prec, Func&& func) {
    if (prec == ov::element::u8) {
        std::forward<Func>(func)(std::integral_constant<ov::element::Type_t, ov::element::u8>{});
    } else if (prec == ov::element::u4) {
        std::forward<Func>(func)(std::integral_constant<ov::element::Type_t, ov::element::u4>{});
    } else {
        std::forward<Func>(func)(std::integral_constant<ov::element::Type_t, ov::element::f32>{});
    }
}

/**
 * @brief Reorder KV cache within a single block or across blocks
 *
 * Handles both non-quantized and by-channel quantized (u8/u4) KV cache.
 * When reordering across blocks with by-channel quantization, performs:
 * dequantize(src) -> copy -> requantize(dst)
 *
 * @tparam PREC Element precision type (u8, u4, or floating point)
 * @param cache PlainTensor for key or value cache [num_blocks, num_heads, block_size, hidden]
 * @param src_block Source block index
 * @param dst_block Destination block index
 * @param src_token Source token index within block
 * @param dst_token Destination token index within block
 * @param head_idx Head index
 * @param block_float_buffer Temporary buffer for dequantized block (size: block_size * hidden)
 * @param row_float_buffer Temporary buffer for dequantized row (size: hidden)
 * @param by_channel Whether by-channel quantization is used
 */
template <ov::element::Type_t PREC>
inline void reorder_kv_cache_token(ov::intel_cpu::PlainTensor& cache,
                                   size_t src_block,
                                   size_t dst_block,
                                   size_t src_token,
                                   size_t dst_token,
                                   size_t head_idx,
                                   std::vector<float>& block_float_buffer,
                                   std::vector<float>& row_float_buffer,
                                   bool by_channel,
                                   size_t dst_block_actual_tokens) {
    const size_t elem_size = cache.get_precision().size();
    const size_t sub_byte = get_sub_byte_multiplier(cache.get_precision());
    const size_t hidden = cache.size(3);
    const size_t params_offset = 2 * sizeof(float) * hidden;

    const size_t token_bytes = hidden * elem_size / sub_byte;

    const size_t data_stride_bytes = cache.stride_bytes(2);

    constexpr bool is_quantized = ov::intel_cpu::any_of(PREC, ov::element::i8, ov::element::u8, ov::element::u4);

    if constexpr (is_quantized) {
        if (by_channel) {
            OPENVINO_ASSERT(src_block != dst_block, "Same-block by-channel should be handled by batch processing");

            auto* src_base = reinterpret_cast<uint8_t*>(cache.ptr_v(src_block, head_idx, 0, 0));
            auto* dst_base = reinterpret_cast<uint8_t*>(cache.ptr_v(dst_block, head_idx, 0, 0));
            auto* src_scales = reinterpret_cast<float*>(src_base);
            auto* src_zps = src_scales + hidden;
            auto* dst_scales = reinterpret_cast<float*>(dst_base);
            auto* dst_zps = dst_scales + hidden;
            auto* src_data = src_base + params_offset;
            auto* dst_data = dst_base + params_offset;

            // Dequantize source row
            attn_dequant_by_channel_kernel<float, PREC>(src_data + src_token * data_stride_bytes,
                                                        row_float_buffer.data(),
                                                        1,
                                                        hidden,
                                                        data_stride_bytes,
                                                        hidden,
                                                        src_scales,
                                                        src_zps);

            // Dequantize destination block (only actual tokens)
            attn_dequant_by_channel_kernel<float, PREC>(dst_data,
                                                        block_float_buffer.data(),
                                                        dst_block_actual_tokens,
                                                        hidden,
                                                        data_stride_bytes,
                                                        hidden,
                                                        dst_scales,
                                                        dst_zps);

            // Copy row into block (in float space)
            std::memcpy(block_float_buffer.data() + dst_token * hidden,
                        row_float_buffer.data(),
                        hidden * sizeof(float));

            // Requantize destination block with updated parameters (only actual tokens)
            quantize_by_channel<float, PREC>(block_float_buffer.data(),
                                             dst_data,
                                             dst_block_actual_tokens,
                                             hidden,
                                             hidden,
                                             data_stride_bytes,
                                             dst_scales,
                                             dst_zps);
        } else {
            auto* src_ptr = reinterpret_cast<uint8_t*>(cache.ptr_v(src_block, head_idx, src_token, 0));
            auto* dst_ptr = reinterpret_cast<uint8_t*>(cache.ptr_v(dst_block, head_idx, dst_token, 0));
            std::memcpy(dst_ptr, src_ptr, token_bytes);
        }
    } else {
        auto* src_ptr = reinterpret_cast<uint8_t*>(cache.ptr_v(src_block, head_idx, src_token, 0));
        auto* dst_ptr = reinterpret_cast<uint8_t*>(cache.ptr_v(dst_block, head_idx, dst_token, 0));
        std::memcpy(dst_ptr, src_ptr, token_bytes);
    }
}

/**
 * @brief Main entry point for KV cache reordering with parallel execution
 *
 * Reorders KV cache based on update indices, supporting both key and value caches
 * with optional by-channel quantization. Parallelizes across (sequence, operation, head) dimensions.
 *
 * Uses thread-local buffers internally for quantization scratch space, eliminating the need
 * for pre-allocated buffers and enabling efficient parallel execution.
 *
 * @param key_cache Key cache tensor [num_blocks, num_heads, block_size, hidden]
 * @param value_cache Value cache tensor [num_blocks, num_heads, block_size, hidden]
 * @param block_indices Physical block indices for each logical block
 * @param block_indices_begins Start indices in block_indices for each sequence
 * @param block_update_indices Pairs of (src, dst) logical token indices to reorder
 * @param block_update_indices_begins Start indices in block_update_indices for each sequence
 * @param key_by_channel Whether key cache uses by-channel quantization
 * @param value_by_channel Whether value cache uses by-channel quantization
 * @param cpu_parallel Parallel execution context for multi-threading
 */
inline void reorder_kv_cache(ov::intel_cpu::PlainTensor& key_cache,
                             ov::intel_cpu::PlainTensor& value_cache,
                             const ov::intel_cpu::PlainTensor& block_indices,
                             const ov::intel_cpu::PlainTensor& block_indices_begins,
                             const ov::intel_cpu::PlainTensor& block_update_indices,
                             const ov::intel_cpu::PlainTensor& block_update_indices_begins,
                             bool key_by_channel,
                             bool value_by_channel,
                             const CpuParallelPtr& cpu_parallel) {
    // Early exit if no updates needed
    if (!block_update_indices || !block_update_indices_begins) {
        return;
    }
    if (block_update_indices.size(0) == 0 || block_update_indices_begins.size(0) == 0) {
        return;
    }

    const size_t kv_heads = key_cache.size(1);
    const size_t key_hidden = key_cache.size(3);
    const size_t value_hidden = value_cache.size(3);

    const size_t key_sub_byte = get_sub_byte_multiplier(key_cache.get_precision());
    const size_t block_size =
        key_by_channel ? (key_cache.size(2) - 2 * sizeof(float) * key_sub_byte) : key_cache.size(2);

    const size_t seq_count = block_update_indices_begins.size(0) == 0 ? 0 : block_update_indices_begins.size(0) - 1;

    const auto* block_idx_ptr = block_indices.ptr<int32_t>();
    const auto* block_idx_begins_ptr = block_indices_begins.ptr<int32_t>();
    const auto* update_ptr = block_update_indices.ptr<int32_t>();
    const auto* update_begins_ptr = block_update_indices_begins.ptr<int32_t>();

    // Cache precision types
    const auto key_prec = key_cache.get_precision();
    const auto value_prec = value_cache.get_precision();

    // For by-channel quantization, operations on the same (block, head) must be sequential
    // because requantization recomputes scale/zp based on entire block statistics.
    // Strategy: parallelize across sequences, keep operations within sequence sequential.

    cpu_parallel->parallel_for(seq_count, [&](size_t seq) {
        // Thread-local buffers for quantization
        thread_local std::vector<float> local_key_block_float;
        thread_local std::vector<float> local_key_row_float;
        thread_local std::vector<float> local_value_block_float;
        thread_local std::vector<float> local_value_row_float;

        if (key_by_channel) {
            local_key_block_float.resize(block_size * key_hidden);
            local_key_row_float.resize(key_hidden);
        }
        if (value_by_channel) {
            local_value_block_float.resize(block_size * value_hidden);
            local_value_row_float.resize(value_hidden);
        }

        const int32_t op_begin = update_begins_ptr[seq];
        const int32_t op_end = update_begins_ptr[seq + 1];

        if (op_end <= op_begin) {
            return;
        }

        const int32_t block_indices_base = block_idx_begins_ptr[seq];
        const int32_t blocks_in_seq = block_idx_begins_ptr[seq + 1] - block_indices_base;

        if (blocks_in_seq <= 0) {
            return;
        }

        int32_t max_src_logical = -1;
        size_t max_src_block_local = 0;
        size_t max_src_token = 0;

        for (int32_t op = op_begin; op < op_end; op++) {
            const int32_t pair_base = op * 2;
            const int32_t src_logical = update_ptr[pair_base + 0];

            if (src_logical > max_src_logical) {
                max_src_logical = src_logical;
                max_src_block_local = static_cast<size_t>(src_logical) / block_size;
                max_src_token = static_cast<size_t>(src_logical) % block_size;
            }
        }

        auto get_block_actual_tokens = [&](size_t block_local) -> size_t {
            if (block_local < max_src_block_local) {
                return block_size;  // Fully filled
            }
            if (block_local == max_src_block_local) {
                return max_src_token + 1;  // Partially filled
            }
            return block_size;
        };

        for (int32_t op = op_begin; op < op_end;) {
            const int32_t pair_base = op * 2;
            const int32_t src_logical = update_ptr[pair_base + 0];
            const int32_t dst_logical = update_ptr[pair_base + 1];

            if (src_logical < 0 || dst_logical < 0) {
                op++;
                continue;
            }

            const size_t src_block_local = static_cast<size_t>(src_logical) / block_size;
            const size_t dst_block_local = static_cast<size_t>(dst_logical) / block_size;
            const size_t src_token = static_cast<size_t>(src_logical) % block_size;
            const size_t dst_token = static_cast<size_t>(dst_logical) % block_size;

            if (src_block_local >= static_cast<size_t>(blocks_in_seq) ||
                dst_block_local >= static_cast<size_t>(blocks_in_seq)) {
                op++;
                continue;
            }

            const auto src_block = static_cast<size_t>(block_idx_ptr[block_indices_base + src_block_local]);
            const auto dst_block = static_cast<size_t>(block_idx_ptr[block_indices_base + dst_block_local]);

            // Check if this is a same-block operation that can be batched
            if (src_block == dst_block && (key_by_channel || value_by_channel)) {
                // Find consecutive same-block operations for this physical block
                int32_t batch_end = op + 1;
                while (batch_end < op_end) {
                    const int32_t next_pair_base = batch_end * 2;
                    const int32_t next_src_logical = update_ptr[next_pair_base + 0];
                    const int32_t next_dst_logical = update_ptr[next_pair_base + 1];

                    if (next_src_logical < 0 || next_dst_logical < 0) {
                        batch_end++;
                        continue;
                    }

                    const size_t next_src_block_local = static_cast<size_t>(next_src_logical) / block_size;
                    const size_t next_dst_block_local = static_cast<size_t>(next_dst_logical) / block_size;

                    if (next_src_block_local >= static_cast<size_t>(blocks_in_seq) ||
                        next_dst_block_local >= static_cast<size_t>(blocks_in_seq)) {
                        batch_end++;
                        continue;
                    }

                    const auto next_src_block =
                        static_cast<size_t>(block_idx_ptr[block_indices_base + next_src_block_local]);
                    const auto next_dst_block =
                        static_cast<size_t>(block_idx_ptr[block_indices_base + next_dst_block_local]);

                    // Stop if not same physical block
                    if (next_src_block != src_block || next_dst_block != dst_block) {
                        break;
                    }

                    batch_end++;
                }

                // Process batched same-block operations for all heads
                const size_t block_actual_tokens = get_block_actual_tokens(src_block_local);

                // Lambda to process a cache with by-channel quantization (batched)
                auto process_cache_by_channel = [&](PlainTensor& cache, ov::element::Type_t prec, size_t hidden_size) {
                    dispatch_precision(prec, [&](auto prec_tag) {
                        constexpr auto PREC = decltype(prec_tag)::value;
                        constexpr bool is_quantized = (PREC == ov::element::u8 || PREC == ov::element::u4);

                        if constexpr (is_quantized) {
                            // Parallel over heads if there are enough heads to benefit from parallelism
                            constexpr size_t HEAD_PARALLEL_THRESHOLD = 16;
                            const bool use_head_parallel = kv_heads >= HEAD_PARALLEL_THRESHOLD;

                            // MSVC requires PREC to be passed explicitly - create wrapper with integral_constant
                            constexpr auto prec_const = std::integral_constant<ov::element::Type_t, PREC>{};

                            auto process_head = [&](size_t h) {
                                constexpr auto P = decltype(prec_const)::value;
                                // Use thread-local buffer for parallel execution
                                thread_local std::vector<float> local_block_buf;
                                local_block_buf.resize(block_actual_tokens * hidden_size);

                                auto* block_base = reinterpret_cast<uint8_t*>(cache.ptr_v(src_block, h, 0, 0));
                                auto* scales = reinterpret_cast<float*>(block_base);
                                auto* zps = scales + hidden_size;
                                auto* data = block_base + 2 * sizeof(float) * hidden_size;
                                const size_t data_stride = cache.stride_bytes(2);

                                // Dequantize once
                                attn_dequant_by_channel_kernel<float, P>(data,
                                                                         local_block_buf.data(),
                                                                         block_actual_tokens,
                                                                         hidden_size,
                                                                         data_stride,
                                                                         hidden_size,
                                                                         scales,
                                                                         zps);

                                // Execute all copy operations in order
                                for (int32_t batch_op = op; batch_op < batch_end; batch_op++) {
                                    const int32_t b_pair_base = batch_op * 2;
                                    const int32_t b_src_logical = update_ptr[b_pair_base + 0];
                                    const int32_t b_dst_logical = update_ptr[b_pair_base + 1];
                                    if (b_src_logical < 0 || b_dst_logical < 0) {
                                        continue;
                                    }

                                    const size_t b_src_token = static_cast<size_t>(b_src_logical) % block_size;
                                    const size_t b_dst_token = static_cast<size_t>(b_dst_logical) % block_size;

                                    std::memcpy(local_block_buf.data() + b_dst_token * hidden_size,
                                                local_block_buf.data() + b_src_token * hidden_size,
                                                hidden_size * sizeof(float));
                                }

                                // Requantize once
                                quantize_by_channel<float, P>(local_block_buf.data(),
                                                              data,
                                                              block_actual_tokens,
                                                              hidden_size,
                                                              hidden_size,
                                                              data_stride,
                                                              scales,
                                                              zps);
                            };

                            if (use_head_parallel) {
                                cpu_parallel->parallel_for(kv_heads, process_head);
                            } else {
                                for (size_t h = 0; h < kv_heads; h++) {
                                    process_head(h);
                                }
                            }
                        }
                    });
                };

                auto process_cache_non_by_channel = [&](PlainTensor& cache,
                                                        ov::element::Type_t prec,
                                                        std::vector<float>& block_float,
                                                        std::vector<float>& row_float) {
                    dispatch_precision(prec, [&](auto prec_tag) {
                        constexpr auto PREC = decltype(prec_tag)::value;

                        for (int32_t batch_op = op; batch_op < batch_end; batch_op++) {
                            const int32_t b_pair_base = batch_op * 2;
                            const int32_t b_src_logical = update_ptr[b_pair_base + 0];
                            const int32_t b_dst_logical = update_ptr[b_pair_base + 1];
                            if (b_src_logical < 0 || b_dst_logical < 0) {
                                continue;
                            }

                            const size_t b_src_token = static_cast<size_t>(b_src_logical) % block_size;
                            const size_t b_dst_token = static_cast<size_t>(b_dst_logical) % block_size;

                            for (size_t h = 0; h < kv_heads; h++) {
                                reorder_kv_cache_token<PREC>(cache,
                                                             src_block,
                                                             src_block,
                                                             b_src_token,
                                                             b_dst_token,
                                                             h,
                                                             block_float,
                                                             row_float,
                                                             false,
                                                             block_actual_tokens);
                            }
                        }
                    });
                };

                if (key_by_channel) {
                    process_cache_by_channel(key_cache, key_prec, key_hidden);
                } else {
                    process_cache_non_by_channel(key_cache, key_prec, local_key_block_float, local_key_row_float);
                }

                if (value_by_channel) {
                    process_cache_by_channel(value_cache, value_prec, value_hidden);
                } else {
                    process_cache_non_by_channel(value_cache,
                                                 value_prec,
                                                 local_value_block_float,
                                                 local_value_row_float);
                }

                // Skip to next unprocessed operation
                op = batch_end;

            } else {
                // Cross-block: process single operation
                const size_t dst_actual_tokens = get_block_actual_tokens(dst_block_local);

                // Lambda to process a single cross-block operation for one cache
                auto process_single_op = [&](PlainTensor& cache, ov::element::Type_t prec, bool by_channel) {
                    dispatch_precision(prec, [&](auto prec_tag) {
                        constexpr auto PREC = decltype(prec_tag)::value;
                        const bool use_by_channel = (PREC == ov::element::u8 || PREC == ov::element::u4) && by_channel;

                        // MSVC requires PREC to be passed explicitly
                        constexpr auto prec_const = std::integral_constant<ov::element::Type_t, PREC>{};

                        auto process_head = [&](size_t h) {
                            constexpr auto P = decltype(prec_const)::value;
                            // Use thread-local buffers for parallel execution
                            thread_local std::vector<float> local_block_buf;
                            thread_local std::vector<float> local_row_buf;

                            const size_t hidden = cache.size(3);
                            local_block_buf.resize(block_size * hidden);
                            local_row_buf.resize(hidden);

                            reorder_kv_cache_token<P>(cache,
                                                      src_block,
                                                      dst_block,
                                                      src_token,
                                                      dst_token,
                                                      h,
                                                      local_block_buf,
                                                      local_row_buf,
                                                      use_by_channel,
                                                      dst_actual_tokens);
                        };

                        cpu_parallel->parallel_for(kv_heads, process_head);
                    });
                };

                process_single_op(key_cache, key_prec, key_by_channel);
                process_single_op(value_cache, value_prec, value_by_channel);

                op++;
            }
        }
    });
}

}  // namespace ov::Extensions::Cpu::XARCH
