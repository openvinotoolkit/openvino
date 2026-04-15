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

using ov::intel_cpu::PlainTensor;
using ov::intel_cpu::CpuParallelPtr;

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
                                   bool by_channel) {
    const size_t hidden = cache.size(3);
    const size_t elem_size = cache.get_precision().size();
    const size_t sub_byte = get_sub_byte_multiplier(cache.get_precision());
    const size_t params_offset = 2 * sizeof(float) * hidden;

    // For by-channel quantization, cache.size(2) includes space for scale/zp params
    // Reference: attn_quant.cpp line 76
    // block_size = dst.m_dims[2] - 2 * sizeof(float) * get_sub_byte_multiplier(DST_PREC)
    const size_t block_size = by_channel
        ? (cache.size(2) - 2 * sizeof(float) * sub_byte)
        : cache.size(2);

    const size_t token_bytes = hidden * elem_size / sub_byte;
    // CRITICAL: Use cache.stride_bytes(2) to match paged attention's stride calculation
    // This accounts for the actual memory layout including any alignment
    const size_t data_stride_bytes = cache.stride_bytes(2);

    if constexpr (ov::intel_cpu::any_of(PREC, ov::element::i8, ov::element::u8, ov::element::u4)) {
        if (by_channel) {
            // By-channel quantization: need to requantize the destination block

            if (src_block != dst_block) {
                // Cross-block case: source and destination use different scale/zp
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

                // Dequantize entire destination block
                attn_dequant_by_channel_kernel<float, PREC>(dst_data,
                                                            block_float_buffer.data(),
                                                            block_size,
                                                            hidden,
                                                            data_stride_bytes,
                                                            hidden,
                                                            dst_scales,
                                                            dst_zps);

                // Copy row into block (in float space)
                std::memcpy(block_float_buffer.data() + dst_token * hidden,
                           row_float_buffer.data(),
                           hidden * sizeof(float));

                // Requantize entire destination block with updated parameters
                quantize_by_channel<float, PREC>(block_float_buffer.data(),
                                                dst_data,
                                                block_size,
                                                hidden,
                                                hidden,
                                                data_stride_bytes,
                                                dst_scales,
                                                dst_zps);
            } else {
                // Same-block case: still need to requantize because data distribution changed
                auto* block_base = reinterpret_cast<uint8_t*>(cache.ptr_v(src_block, head_idx, 0, 0));
                auto* scales = reinterpret_cast<float*>(block_base);
                auto* zps = scales + hidden;
                auto* data = block_base + params_offset;

                // Dequantize entire block
                attn_dequant_by_channel_kernel<float, PREC>(data,
                                                            block_float_buffer.data(),
                                                            block_size,
                                                            hidden,
                                                            data_stride_bytes,
                                                            hidden,
                                                            scales,
                                                            zps);

                // Move token in float space
                std::memcpy(block_float_buffer.data() + dst_token * hidden,
                           block_float_buffer.data() + src_token * hidden,
                           hidden * sizeof(float));

                // Requantize entire block with new parameters (based on updated distribution)
                quantize_by_channel<float, PREC>(block_float_buffer.data(),
                                                data,
                                                block_size,
                                                hidden,
                                                hidden,
                                                data_stride_bytes,
                                                scales,
                                                zps);
            }
        } else {
            // Non by-channel quantization: direct copy of quantized data
            auto* src_ptr = reinterpret_cast<uint8_t*>(cache.ptr_v(src_block, head_idx, src_token, 0));
            auto* dst_ptr = reinterpret_cast<uint8_t*>(cache.ptr_v(dst_block, head_idx, dst_token, 0));
            std::memcpy(dst_ptr, src_ptr, token_bytes);
        }
    } else {
        // Non-quantized types: direct memcpy
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

    // Calculate actual block size (excluding scale/zp params for by-channel quantization)
    // For by-channel: cache.size(2) includes params space, need to subtract it
    // Reference: attn_quant.cpp line 76
    const size_t key_elem_size = key_cache.get_precision().size();
    const size_t key_sub_byte = get_sub_byte_multiplier(key_cache.get_precision());
    const size_t block_size = key_by_channel
        ? (key_cache.size(2) - 2 * sizeof(float) * key_sub_byte)
        : key_cache.size(2);

    const size_t seq_count =
        block_update_indices_begins.size(0) == 0 ? 0 : block_update_indices_begins.size(0) - 1;

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

        // Allocate buffers only if quantization is used
        if (key_by_channel) {
            if (local_key_block_float.size() != block_size * key_hidden) {
                local_key_block_float.resize(block_size * key_hidden);
                local_key_row_float.resize(key_hidden);
            }
        }
        if (value_by_channel) {
            if (local_value_block_float.size() != block_size * value_hidden) {
                local_value_block_float.resize(block_size * value_hidden);
                local_value_row_float.resize(value_hidden);
            }
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

        // Sequential processing of operations within each sequence to maintain correctness
        for (int32_t op = op_begin; op < op_end; op++) {
            const int32_t pair_base = op * 2;
            const int32_t src_logical = update_ptr[pair_base + 0];
            const int32_t dst_logical = update_ptr[pair_base + 1];

            if (src_logical < 0 || dst_logical < 0) {
                continue;
            }

            const size_t src_block_local = static_cast<size_t>(src_logical) / block_size;
            const size_t dst_block_local = static_cast<size_t>(dst_logical) / block_size;
            const size_t src_token = static_cast<size_t>(src_logical) % block_size;
            const size_t dst_token = static_cast<size_t>(dst_logical) % block_size;

            if (src_block_local >= static_cast<size_t>(blocks_in_seq) ||
                dst_block_local >= static_cast<size_t>(blocks_in_seq)) {
                continue;
            }

            const size_t src_block = static_cast<size_t>(block_idx_ptr[block_indices_base + src_block_local]);
            const size_t dst_block = static_cast<size_t>(block_idx_ptr[block_indices_base + dst_block_local]);

            // Process all heads for this operation
            for (size_t h = 0; h < kv_heads; h++) {
                // Process key cache
                if (key_prec == ov::element::u8) {
                    reorder_kv_cache_token<ov::element::u8>(key_cache,
                                                            src_block,
                                                            dst_block,
                                                            src_token,
                                                            dst_token,
                                                            h,
                                                            local_key_block_float,
                                                            local_key_row_float,
                                                            key_by_channel);
                } else if (key_prec == ov::element::u4) {
                    reorder_kv_cache_token<ov::element::u4>(key_cache,
                                                            src_block,
                                                            dst_block,
                                                            src_token,
                                                            dst_token,
                                                            h,
                                                            local_key_block_float,
                                                            local_key_row_float,
                                                            key_by_channel);
                } else {
                    reorder_kv_cache_token<ov::element::f32>(key_cache,
                                                             src_block,
                                                             dst_block,
                                                             src_token,
                                                             dst_token,
                                                             h,
                                                             local_key_block_float,
                                                             local_key_row_float,
                                                             false);
                }

                // Process value cache
                if (value_prec == ov::element::u8) {
                    reorder_kv_cache_token<ov::element::u8>(value_cache,
                                                            src_block,
                                                            dst_block,
                                                            src_token,
                                                            dst_token,
                                                            h,
                                                            local_value_block_float,
                                                            local_value_row_float,
                                                            value_by_channel);
                } else if (value_prec == ov::element::u4) {
                    reorder_kv_cache_token<ov::element::u4>(value_cache,
                                                            src_block,
                                                            dst_block,
                                                            src_token,
                                                            dst_token,
                                                            h,
                                                            local_value_block_float,
                                                            local_value_row_float,
                                                            value_by_channel);
                } else {
                    reorder_kv_cache_token<ov::element::f32>(value_cache,
                                                             src_block,
                                                             dst_block,
                                                             src_token,
                                                             dst_token,
                                                             h,
                                                             local_value_block_float,
                                                             local_value_row_float,
                                                             false);
                }
            }
        }
    });
}

}  // namespace ov::Extensions::Cpu::XARCH
