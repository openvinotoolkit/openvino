// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "cache_reorder.hpp"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

#include "attn_quant_kernel.hpp"
#include "common.hpp"
#include "cpu_parallel.hpp"
#include "openvino/core/except.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/core/type/element_type_traits.hpp"
#include "utils/plain_tensor.hpp"
namespace ov::Extensions::Cpu::XARCH {

using ov::intel_cpu::CpuParallelPtr;
using ov::intel_cpu::PlainTensor;

namespace {

/**
 * @brief Batch reorder context for collecting operations with same block pair
 */
struct BatchReorderContext {
    size_t src_block = 0;
    size_t dst_block = 0;
    bool same_block = false;
    int32_t op_begin = 0;
    int32_t op_end = 0;
    size_t block_size = 0;
    size_t dst_actual_tokens = 0;
    const int32_t* update_ptr = nullptr;
    size_t hidden = 0;
};

/**
 * @brief Process cache batch for quantized types
 */
template <ov::element::Type_t PREC>
void process_cache_batch_quantized(PlainTensor& cache,
                                   const BatchReorderContext& ctx,
                                   bool by_channel,
                                   size_t kv_heads,
                                   const CpuParallelPtr& cpu_parallel) {
    using value_type = typename ov::element_type_traits<PREC>::value_type;
    static constexpr size_t elem_size = sizeof(value_type);
    static constexpr size_t sub_byte = get_sub_byte_multiplier(PREC);
    const size_t token_bytes = ctx.hidden * elem_size / sub_byte;

    auto process_head = [&](size_t h) {
        if (by_channel) {
            thread_local std::vector<float> local_block_buf;
            thread_local std::vector<float> local_row_buf;
            local_block_buf.resize(ctx.block_size * ctx.hidden);
            local_row_buf.resize(ctx.hidden);

            auto* src_base = reinterpret_cast<uint8_t*>(cache.ptr_v(ctx.src_block, h, 0, 0));
            auto* dst_base = reinterpret_cast<uint8_t*>(cache.ptr_v(ctx.dst_block, h, 0, 0));
            auto* src_scales = reinterpret_cast<float*>(src_base);
            auto* src_zps = src_scales + ctx.hidden;
            auto* dst_scales = reinterpret_cast<float*>(dst_base);
            auto* dst_zps = dst_scales + ctx.hidden;

            const size_t params_offset = 2 * sizeof(float) * ctx.hidden;
            auto* src_data = src_base + params_offset;
            auto* dst_data = dst_base + params_offset;
            const size_t data_stride = cache.stride_bytes(2);

            if (ctx.same_block) {
                for (int32_t batch_op = ctx.op_begin; batch_op < ctx.op_end; batch_op++) {
                    const int32_t b_pair_base = batch_op * 2;
                    const int32_t b_src_logical = ctx.update_ptr[b_pair_base + 0];
                    const int32_t b_dst_logical = ctx.update_ptr[b_pair_base + 1];
                    if (b_src_logical < 0 || b_dst_logical < 0) {
                        continue;
                    }

                    const auto b_src_token = static_cast<size_t>(b_src_logical) % ctx.block_size;
                    const auto b_dst_token = static_cast<size_t>(b_dst_logical) % ctx.block_size;

                    std::memcpy(dst_data + b_dst_token * data_stride,
                                src_data + b_src_token * data_stride,
                                token_bytes);
                }
            } else {
                attn_dequant_by_channel_kernel<float, PREC>(dst_data,
                                                            local_block_buf.data(),
                                                            ctx.dst_actual_tokens,
                                                            ctx.hidden,
                                                            data_stride,
                                                            ctx.hidden,
                                                            dst_scales,
                                                            dst_zps);

                for (int32_t batch_op = ctx.op_begin; batch_op < ctx.op_end; batch_op++) {
                    const int32_t b_pair_base = batch_op * 2;
                    const int32_t b_src_logical = ctx.update_ptr[b_pair_base + 0];
                    const int32_t b_dst_logical = ctx.update_ptr[b_pair_base + 1];
                    if (b_src_logical < 0 || b_dst_logical < 0) {
                        continue;
                    }

                    const auto b_src_token = static_cast<size_t>(b_src_logical) % ctx.block_size;
                    const auto b_dst_token = static_cast<size_t>(b_dst_logical) % ctx.block_size;

                    attn_dequant_by_channel_kernel<float, PREC>(src_data + b_src_token * data_stride,
                                                                local_row_buf.data(),
                                                                1,
                                                                ctx.hidden,
                                                                data_stride,
                                                                ctx.hidden,
                                                                src_scales,
                                                                src_zps);

                    std::memcpy(local_block_buf.data() + b_dst_token * ctx.hidden,
                                local_row_buf.data(),
                                ctx.hidden * sizeof(float));
                }

                quantize_by_channel<float, PREC>(local_block_buf.data(),
                                                 dst_data,
                                                 ctx.dst_actual_tokens,
                                                 ctx.hidden,
                                                 ctx.hidden,
                                                 data_stride,
                                                 dst_scales,
                                                 dst_zps);
            }
        } else {
            for (int32_t batch_op = ctx.op_begin; batch_op < ctx.op_end; batch_op++) {
                const int32_t b_pair_base = batch_op * 2;
                const int32_t b_src_logical = ctx.update_ptr[b_pair_base + 0];
                const int32_t b_dst_logical = ctx.update_ptr[b_pair_base + 1];
                if (b_src_logical < 0 || b_dst_logical < 0) {
                    continue;
                }

                const auto b_src_token = static_cast<size_t>(b_src_logical) % ctx.block_size;
                const auto b_dst_token = static_cast<size_t>(b_dst_logical) % ctx.block_size;

                auto* src_ptr = reinterpret_cast<uint8_t*>(cache.ptr_v(ctx.src_block, h, b_src_token, 0));
                auto* dst_ptr = reinterpret_cast<uint8_t*>(cache.ptr_v(ctx.dst_block, h, b_dst_token, 0));
                std::memcpy(dst_ptr, src_ptr, token_bytes);
            }
        }
    };

    cpu_parallel->parallel_for(kv_heads, process_head);
}

/**
 * @brief Process cache batch for non-quantized types
 */
template <ov::element::Type_t PREC>
void process_cache_batch_non_quantized(PlainTensor& cache,
                                       const BatchReorderContext& ctx,
                                       size_t kv_heads,
                                       const CpuParallelPtr& cpu_parallel) {
    using value_type = typename ov::element_type_traits<PREC>::value_type;
    static constexpr size_t elem_size = sizeof(value_type);
    static constexpr size_t sub_byte = get_sub_byte_multiplier(PREC);
    const size_t token_bytes = ctx.hidden * elem_size / sub_byte;

    auto process_head = [&](size_t h) {
        for (int32_t batch_op = ctx.op_begin; batch_op < ctx.op_end; batch_op++) {
            const int32_t b_pair_base = batch_op * 2;
            const int32_t b_src_logical = ctx.update_ptr[b_pair_base + 0];
            const int32_t b_dst_logical = ctx.update_ptr[b_pair_base + 1];
            if (b_src_logical < 0 || b_dst_logical < 0) {
                continue;
            }

            const auto b_src_token = static_cast<size_t>(b_src_logical) % ctx.block_size;
            const auto b_dst_token = static_cast<size_t>(b_dst_logical) % ctx.block_size;

            auto* src_ptr = reinterpret_cast<uint8_t*>(cache.ptr_v(ctx.src_block, h, b_src_token, 0));
            auto* dst_ptr = reinterpret_cast<uint8_t*>(cache.ptr_v(ctx.dst_block, h, b_dst_token, 0));
            std::memcpy(dst_ptr, src_ptr, token_bytes);
        }
    };

    cpu_parallel->parallel_for(kv_heads, process_head);
}

void dispatch_and_process_cache(ov::element::Type_t prec,
                                PlainTensor& cache,
                                const BatchReorderContext& ctx,
                                bool by_channel,
                                size_t kv_heads,
                                const CpuParallelPtr& cpu_parallel) {
    if (prec == ov::element::u8) {
        process_cache_batch_quantized<ov::element::u8>(cache, ctx, by_channel, kv_heads, cpu_parallel);
    } else if (prec == ov::element::u4) {
        process_cache_batch_quantized<ov::element::u4>(cache, ctx, by_channel, kv_heads, cpu_parallel);
    } else if (prec == ov::element::f32) {
        process_cache_batch_non_quantized<ov::element::f32>(cache, ctx, kv_heads, cpu_parallel);
    } else if (prec == ov::element::bf16) {
        process_cache_batch_non_quantized<ov::element::bf16>(cache, ctx, kv_heads, cpu_parallel);
    } else if (prec == ov::element::f16) {
        process_cache_batch_non_quantized<ov::element::f16>(cache, ctx, kv_heads, cpu_parallel);
    } else {
        OPENVINO_THROW("PaKVReorder supports only f32, bf16, f16, u8, and u4 KV cache precisions. ",
                       "int8 KV cache is not supported.");
    }
}

}  // namespace

void reorder_kv_cache(PlainTensor& key_cache,
                      PlainTensor& value_cache,
                      const PlainTensor& block_indices,
                      const PlainTensor& block_indices_begins,
                      const PlainTensor& block_update_indices,
                      const PlainTensor& block_update_indices_begins,
                      bool key_by_channel,
                      bool value_by_channel,
                      const CpuParallelPtr& cpu_parallel) {
    if (!block_update_indices || !block_update_indices_begins || block_update_indices.size(0) == 0 ||
        block_update_indices_begins.size(0) == 0) {
        return;
    }

    const size_t kv_heads = key_cache.size(1);
    const size_t key_hidden = key_cache.size(3);
    const size_t value_hidden = value_cache.size(3);
    const size_t key_sub_byte = get_sub_byte_multiplier(key_cache.get_precision());
    const size_t block_size =
        key_by_channel ? (key_cache.size(2) - 2 * sizeof(float) * key_sub_byte) : key_cache.size(2);
    const size_t seq_count = block_update_indices_begins.size(0) - 1;

    const auto* block_idx_ptr = block_indices.ptr<int32_t>();
    const auto* block_idx_begins_ptr = block_indices_begins.ptr<int32_t>();
    const auto* update_ptr = block_update_indices.ptr<int32_t>();
    const auto* update_begins_ptr = block_update_indices_begins.ptr<int32_t>();

    const auto key_prec = key_cache.get_precision();
    const auto value_prec = value_cache.get_precision();

    cpu_parallel->parallel_for(seq_count, [&](size_t seq) {
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

        int32_t max_dst_logical = -1;
        size_t max_dst_block_local = 0;
        size_t max_dst_token = 0;
        for (int32_t op = op_begin; op < op_end; op++) {
            const int32_t dst_logical = update_ptr[op * 2 + 1];
            if (dst_logical > max_dst_logical) {
                max_dst_logical = dst_logical;
                max_dst_block_local = static_cast<size_t>(dst_logical) / block_size;
                max_dst_token = static_cast<size_t>(dst_logical) % block_size;
            }
        }

        auto get_block_actual_tokens = [&](size_t block_local) -> size_t {
            if (block_local < max_dst_block_local) {
                return block_size;
            }
            if (block_local == max_dst_block_local) {
                return max_dst_token + 1;
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

            const auto src_block_local = static_cast<size_t>(src_logical) / block_size;
            const auto dst_block_local = static_cast<size_t>(dst_logical) / block_size;

            if (src_block_local >= static_cast<size_t>(blocks_in_seq) ||
                dst_block_local >= static_cast<size_t>(blocks_in_seq)) {
                op++;
                continue;
            }

            const auto src_block = static_cast<size_t>(block_idx_ptr[block_indices_base + src_block_local]);
            const auto dst_block = static_cast<size_t>(block_idx_ptr[block_indices_base + dst_block_local]);

            int32_t batch_end = op + 1;
            while (batch_end < op_end) {
                const int32_t n_pair_base = batch_end * 2;
                const int32_t n_src_logical = update_ptr[n_pair_base + 0];
                const int32_t n_dst_logical = update_ptr[n_pair_base + 1];

                if (n_src_logical < 0 || n_dst_logical < 0) {
                    batch_end++;
                    continue;
                }

                const auto n_src_block_local = static_cast<size_t>(n_src_logical) / block_size;
                const auto n_dst_block_local = static_cast<size_t>(n_dst_logical) / block_size;

                if (n_src_block_local >= static_cast<size_t>(blocks_in_seq) ||
                    n_dst_block_local >= static_cast<size_t>(blocks_in_seq)) {
                    batch_end++;
                    continue;
                }

                const auto n_src_block = static_cast<size_t>(block_idx_ptr[block_indices_base + n_src_block_local]);
                const auto n_dst_block = static_cast<size_t>(block_idx_ptr[block_indices_base + n_dst_block_local]);

                if (n_src_block != src_block || n_dst_block != dst_block) {
                    break;
                }
                batch_end++;
            }

            BatchReorderContext key_ctx;
            key_ctx.src_block = src_block;
            key_ctx.dst_block = dst_block;
            key_ctx.same_block = (src_block == dst_block);
            key_ctx.op_begin = op;
            key_ctx.op_end = batch_end;
            key_ctx.block_size = block_size;
            key_ctx.dst_actual_tokens = get_block_actual_tokens(dst_block_local);
            key_ctx.update_ptr = update_ptr;
            key_ctx.hidden = key_hidden;

            dispatch_and_process_cache(key_prec, key_cache, key_ctx, key_by_channel, kv_heads, cpu_parallel);

            BatchReorderContext value_ctx;
            value_ctx.src_block = src_block;
            value_ctx.dst_block = dst_block;
            value_ctx.same_block = (src_block == dst_block);
            value_ctx.op_begin = op;
            value_ctx.op_end = batch_end;
            value_ctx.block_size = block_size;
            value_ctx.dst_actual_tokens = get_block_actual_tokens(dst_block_local);
            value_ctx.update_ptr = update_ptr;
            value_ctx.hidden = value_hidden;

            dispatch_and_process_cache(value_prec, value_cache, value_ctx, value_by_channel, kv_heads, cpu_parallel);

            op = batch_end;
        }
    });
}

}  // namespace ov::Extensions::Cpu::XARCH
