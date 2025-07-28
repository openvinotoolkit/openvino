// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <type_traits>

#include "openvino/core/except.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/core/type/float16.hpp"
#include "utils/general_utils.h"
#include "utils/plain_tensor.hpp"

#if defined(HAVE_AVX2) || defined(HAVE_AVX512F)
#    include <immintrin.h>
#endif

#include "attn_quant.hpp"
#include "attn_quant_kernel.hpp"
#include "nodes/kernels/scaled_attn/common.hpp"
#include "openvino/core/parallel.hpp"
#include "openvino/core/type/bfloat16.hpp"

namespace ov::Extensions::Cpu::XARCH {

using namespace ov;

template <typename T, ov::element::Type_t DST_PREC>
static void quantize_block_by_dims(const ov::intel_cpu::PlainTensor& src,
                                   const ov::intel_cpu::PlainTensor& dst,
                                   size_t b,
                                   size_t h,
                                   size_t m,
                                   size_t block_number,
                                   size_t block_offset,
                                   size_t group_size) {
    // The cache layout is [scale0, zp0]|[group0]|[scale1, zp1]|[group1].....
    // dst_offset is the offset among groups. The addition of 2 * sizeof(float) aims to shift to next group
    // base pointer points to the base address of next group.
    // base +  2 * sizeof(float) aims to skip the scale/zp within the group.
    constexpr size_t sub_byte_multiplier = DST_PREC == ov::element::u4 ? 2 : 1;
    size_t S = src.m_dims[3];
    constexpr size_t param_size = sizeof(float) * (DST_PREC == ov::element::i8 ? 1 : 2);
    for (size_t src_offset = 0, dst_offset = 0; src_offset < S;
         src_offset += group_size, dst_offset += group_size / sub_byte_multiplier + param_size) {
        auto base = dst.ptr<uint8_t, DST_PREC>(block_number, h, block_offset, 0);
        base += dst_offset;
        auto p = reinterpret_cast<float*>(base);
        uint8_t* ptr = base + param_size;
        quantize<T, DST_PREC>(src.ptr<T>(b, h, m, src_offset), ptr, group_size, p);
    }
}

template <typename T, ov::element::Type_t DST_PREC>
static void quantize_block_by_channel(const ov::intel_cpu::PlainTensor& src,
                                      const ov::intel_cpu::PlainTensor& dst,
                                      const ov::intel_cpu::PlainTensor& past_lens,
                                      const ov::intel_cpu::PlainTensor& subsequence_begins,
                                      const ov::intel_cpu::PlainTensor& block_indices,
                                      const ov::intel_cpu::PlainTensor& block_indices_begins,
                                      ov::intel_cpu::PlainTensor& temp_buffer,
                                      size_t sub_seq_id,
                                      size_t h) {
    // scale f32[S] zp f32[S] offset in bytes
    const auto S = src.m_dims[3];
    const size_t params_offset = 2 * sizeof(float) * S;
    const auto past_len = past_lens.ptr<int32_t>()[sub_seq_id];
    const auto q_len =
        subsequence_begins.ptr<int32_t>()[sub_seq_id + 1] - subsequence_begins.ptr<int32_t>()[sub_seq_id];
    const auto block_number_start = block_indices_begins.ptr<int32_t>()[sub_seq_id];
    const size_t block_size = dst.m_dims[2] - 2 * sizeof(float) * get_sub_byte_multiplier(DST_PREC);
    const size_t m = 0;
    // Quantized cache is either u8/u4, the plain memory is both uint8,
    // Here we use stride_bytes instead of stride which consider divide sub_byte_multiplier automatically.
    if (past_len == 0) {
        // (block_indices_begins[sub_seq_id + 1] - 1) is not a reliable upper bound for sub_seq,
        // (block_indices_begins[sub_seq_id + 1] - 1) may not exist in current batch under vLLM case,
        // For example prompt_size=391, max-num-batched-tokens=256, vLLM allocates 13 blocks [0, 13)
        // but in first iteration, vLLM only feeds 256 tokens, [0,8).
        const auto total_blocks = ov::intel_cpu::div_up(q_len, block_size);
        parallel_for(total_blocks, [&](int32_t block_count) {
            const auto block_id = block_number_start + block_count;
            const auto block_number = block_indices.ptr<int32_t>()[block_id];
            const auto token_num =
                (block_count == (total_blocks - 1)) ? (q_len - block_count * block_size) : block_size;
            const size_t b_in_tokens = subsequence_begins.ptr<int32_t>()[sub_seq_id] + block_count * block_size;
            auto base = dst.ptr<uint8_t, DST_PREC>(block_number, h, 0, 0);
            auto p_scales = reinterpret_cast<float*>(base);
            auto p_zps = p_scales + S;
            auto p_data = base + params_offset;
            quantize_by_channel<T, DST_PREC>(src.ptr<T>(b_in_tokens, h, m),
                                             p_data,
                                             token_num,
                                             S,
                                             src.stride(0),
                                             dst.stride_bytes(2),
                                             p_scales,
                                             p_zps);
        });
    } else {
        const auto prev_nums = past_len % block_size;
        const size_t block_offset = block_number_start + past_len / block_size;
        const auto total_blocks = ov::intel_cpu::div_up(prev_nums + q_len, block_size);
        parallel_for(total_blocks, [&](size_t block_id) {
            const bool is_first_block = block_id == 0;
            size_t offset = 0;
            // layout for blocked cache
            // block_0    |   block_1  |   block_2
            // prev_data  |   new_data |   new_data
            // new_data   |   new_data |   new_data
            if (!is_first_block) {
                offset = prev_nums;
            }
            const size_t b_in_tokens = subsequence_begins.ptr<int32_t>()[sub_seq_id] + block_size * block_id - offset;
            const auto block_number = block_indices.ptr<int32_t>()[block_id + block_offset];
            auto base = dst.ptr<uint8_t, DST_PREC>(block_number, h, 0, 0);
            auto p_scales = reinterpret_cast<float*>(base);
            auto p_zps = p_scales + S;
            auto p_data = base + params_offset;
            size_t valid_length = 0;
            float* buffer = temp_buffer.ptr<float>(parallel_get_thread_num());
            if (is_first_block) {
                valid_length = std::min(static_cast<size_t>(q_len), block_size - prev_nums);
            } else {
                // first block may have pre-filled data, the offset of first block is prev_nums, following
                // blocks have offset = block_size
                valid_length = std::min(static_cast<size_t>(q_len) + prev_nums - block_size * block_id, block_size);
            }
            if (is_first_block && prev_nums) {
                attn_dequant_by_channel_kernel<float, DST_PREC>(p_data,
                                                                buffer,
                                                                prev_nums,
                                                                S,
                                                                dst.stride_bytes(2),
                                                                S,
                                                                p_scales,
                                                                p_zps);
                cvt_copy(buffer + prev_nums * S, src.ptr<T>(b_in_tokens, h, m), valid_length, S, src.stride(0), S);
                quantize_by_channel<float, DST_PREC>(buffer,
                                                     p_data,
                                                     prev_nums + valid_length,
                                                     S,
                                                     S,
                                                     dst.stride_bytes(2),
                                                     p_scales,
                                                     p_zps);
            } else {
                quantize_by_channel<T, DST_PREC>(src.ptr<T>(b_in_tokens, h, m),
                                                 p_data,
                                                 valid_length,
                                                 S,
                                                 src.stride(0),
                                                 dst.stride_bytes(2),
                                                 p_scales,
                                                 p_zps);
            }
        });
    }
}

template <typename T, typename T2>
static void attn_quant_mt(const ov::intel_cpu::PlainTensor& k_src,
                          const ov::intel_cpu::PlainTensor& v_src,
                          const ov::intel_cpu::PlainTensor& k_dst,
                          const ov::intel_cpu::PlainTensor& v_dst,
                          const size_t L0,
                          float* temp_buffer,
                          const ov::intel_cpu::PlainTensor& k_scale_zp,
                          const ov::intel_cpu::PlainTensor& v_scale_zp,
                          const bool quant_key_by_channel,
                          const size_t key_group_size,
                          const size_t value_group_size) {
    // For compatibility, all input_kvs are permuted to BHLS
    size_t B = k_src.m_dims[0];
    size_t H = k_src.m_dims[1];
    size_t L1 = k_src.m_dims[2];
    size_t S = k_src.m_dims[3];
    size_t SV = v_src.m_dims[3];
    if (quant_key_by_channel) {
        if (L0 == 0) {
            parallel_for3d(ov::intel_cpu::div_up(L1, key_group_size), B, H, [&](size_t group_id, size_t b, size_t h) {
                quantize_by_channel<T, intel_cpu::precision_of<T2>::value>(
                    k_src.ptr<T>(b, h, group_id * key_group_size),
                    k_dst.ptr<T2>(b, h, group_id * key_group_size),
                    std::min(key_group_size, L1 - group_id * key_group_size),
                    S,
                    k_src.m_strides[2],
                    k_dst.m_strides[2],
                    k_scale_zp.ptr<float>(group_id * 2, b, h),
                    k_scale_zp.ptr<float>(group_id * 2 + 1, b, h));
            });
        } else {
            size_t group_id = L0 / key_group_size;
            size_t prev_nums = L0 % key_group_size;
            parallel_for2d(B, H, [&](size_t b, size_t h) {
                auto thread_id = parallel_get_thread_num();
                float* thread_temp_buffer = temp_buffer + thread_id * key_group_size * S;
                size_t remaining_group_size = prev_nums ? (key_group_size - prev_nums) : 0;
                if (prev_nums) {
                    attn_dequant_by_channel_kernel<float, intel_cpu::precision_of<T2>::value>(
                        k_dst.ptr<uint8_t>(b, h, group_id * key_group_size),
                        thread_temp_buffer,
                        prev_nums,
                        S,
                        k_dst.stride_bytes(2),
                        S,
                        k_scale_zp.ptr<float>(group_id * 2, b, h),
                        k_scale_zp.ptr<float>(group_id * 2 + 1, b, h));
                    remaining_group_size = std::min(remaining_group_size, L1);
                    cvt_copy(thread_temp_buffer + prev_nums * S,
                             k_src.ptr<T>(b, h),
                             remaining_group_size,
                             S,
                             k_src.m_strides[2],
                             S);
                    quantize_by_channel<float, intel_cpu::precision_of<T2>::value>(
                        thread_temp_buffer,
                        k_dst.ptr<T2>(b, h, group_id * key_group_size),
                        remaining_group_size + prev_nums,
                        S,
                        S,
                        k_dst.m_strides[2],
                        k_scale_zp.ptr<float>(group_id * 2, b, h),
                        k_scale_zp.ptr<float>(group_id * 2 + 1, b, h));
                }

                if (L1 > remaining_group_size) {
                    size_t new_seq = L1 - remaining_group_size;
                    for (size_t new_group_id = prev_nums ? group_id + 1 : group_id, src_offset = 0;
                         new_group_id < ov::intel_cpu::div_up(L0 + L1, key_group_size);
                         new_group_id++, src_offset += key_group_size) {
                        quantize_by_channel<T, intel_cpu::precision_of<T2>::value>(
                            k_src.ptr<T>(b, h, remaining_group_size + src_offset),
                            k_dst.ptr<T2>(b, h, new_group_id * key_group_size),
                            std::min(key_group_size, new_seq - src_offset),
                            S,
                            k_src.m_strides[2],
                            k_dst.m_strides[2],
                            k_scale_zp.ptr<float>(new_group_id * 2, b, h),
                            k_scale_zp.ptr<float>(new_group_id * 2 + 1, b, h));
                    }
                }
            });
        }
    } else {
        parallel_for3d(L1, B, H, [&](size_t m, size_t b, size_t h) {
            auto* p_k = k_scale_zp.ptr<float>(L0 + m, b, h);
            for (size_t group_id = 0; group_id < S / key_group_size; group_id++) {
                quant_u8(k_src.ptr<T>(b, h, m, group_id * key_group_size),
                         k_dst.ptr<T2>(b, h, L0 + m, group_id * key_group_size),
                         key_group_size,
                         p_k[group_id * 2],
                         p_k[group_id * 2 + 1]);
            }
        });
    }
    parallel_for3d(L1, B, H, [&](size_t m, size_t b, size_t h) {
        auto* p_v = v_scale_zp.ptr<float>(L0 + m, b, h);
        for (size_t group_id = 0; group_id < SV / value_group_size; group_id++) {
            quant_u8(v_src.ptr<T>(b, h, m, group_id * value_group_size),
                     v_dst.ptr<T2>(b, h, L0 + m, group_id * value_group_size),
                     value_group_size,
                     p_v[group_id * 2],
                     p_v[group_id * 2 + 1]);
        }
    });
}

template <typename T, ov::element::Type_t KEY_DST_PREC, ov::element::Type_t VALUE_DST_PREC>
static void saged_attn_quant_mt(const ov::intel_cpu::PlainTensor& k_src,
                                const ov::intel_cpu::PlainTensor& v_src,
                                const ov::intel_cpu::PlainTensor& k_dst,
                                const ov::intel_cpu::PlainTensor& v_dst,
                                const ov::intel_cpu::PlainTensor& past_lens,
                                const ov::intel_cpu::PlainTensor& subsequence_begins,
                                const ov::intel_cpu::PlainTensor& block_indices,
                                const ov::intel_cpu::PlainTensor& block_indices_begins,
                                const ov::intel_cpu::PlainTensor& slot_mapping,
                                ov::intel_cpu::PlainTensor& temp_buffer,
                                const QuantizeParam& quant_param) {
    const size_t B = k_src.m_dims[0], H = k_src.m_dims[1], L1 = k_src.m_dims[2];
    const size_t block_size = quant_param.quant_key_by_channel
                                  ? k_dst.m_dims[2] - 2 * sizeof(float) * get_sub_byte_multiplier(KEY_DST_PREC)
                                  : k_dst.m_dims[2];
    // quant key
    parallel_for3d(B, L1, H, [&](size_t b, size_t m, size_t h) {
        auto slot = slot_mapping.ptr<int32_t>(b)[m];
        if (slot < 0) {
            return;
        }
        auto block_number = slot / block_size;
        auto block_offset = slot % block_size;
        quantize_block_by_dims<T, KEY_DST_PREC>(k_src,
                                                k_dst,
                                                b,
                                                h,
                                                m,
                                                block_number,
                                                block_offset,
                                                quant_param.key_group_size);

        quantize_block_by_dims<T, VALUE_DST_PREC>(v_src,
                                                  v_dst,
                                                  b,
                                                  h,
                                                  m,
                                                  block_number,
                                                  block_offset,
                                                  quant_param.value_group_size);
    });
}

template <typename T, ov::element::Type_t KEY_DST_PREC, ov::element::Type_t VALUE_DST_PREC>
static void paged_attn_quant_mt(const ov::intel_cpu::PlainTensor& k_src,
                                const ov::intel_cpu::PlainTensor& v_src,
                                const ov::intel_cpu::PlainTensor& k_dst,
                                const ov::intel_cpu::PlainTensor& v_dst,
                                const ov::intel_cpu::PlainTensor& past_lens,
                                const ov::intel_cpu::PlainTensor& subsequence_begins,
                                const ov::intel_cpu::PlainTensor& block_indices,
                                const ov::intel_cpu::PlainTensor& block_indices_begins,
                                const ov::intel_cpu::PlainTensor& slot_mapping,
                                ov::intel_cpu::PlainTensor& temp_buffer,
                                const QuantizeParam& quant_param) {
    const size_t B = k_src.m_dims[0], H = k_src.m_dims[1], L1 = k_src.m_dims[2];
    const size_t block_size = quant_param.quant_key_by_channel
                                  ? k_dst.m_dims[2] - 2 * sizeof(float) * get_sub_byte_multiplier(KEY_DST_PREC)
                                  : k_dst.m_dims[2];
    if (quant_param.quant_key_by_channel) {
        parallel_for2d(past_lens.size(0), H, [&](size_t sub_seq_id, size_t h) {
            quantize_block_by_channel<T, KEY_DST_PREC>(k_src,
                                                       k_dst,
                                                       past_lens,
                                                       subsequence_begins,
                                                       block_indices,
                                                       block_indices_begins,
                                                       temp_buffer,
                                                       sub_seq_id,
                                                       h);
        });
    } else {
        parallel_for3d(B, L1, H, [&](size_t b, size_t m, size_t h) {
            auto slot = slot_mapping.ptr<int32_t>(b)[m];
            if (slot < 0) {
                return;
            }
            auto block_number = slot / block_size;
            auto block_offset = slot % block_size;
            quantize_block_by_dims<T, KEY_DST_PREC>(k_src,
                                                    k_dst,
                                                    b,
                                                    h,
                                                    m,
                                                    block_number,
                                                    block_offset,
                                                    quant_param.key_group_size);
        });
    }
    // quant value
    if (quant_param.quant_value_by_channel) {
        parallel_for2d(past_lens.size(0), H, [&](size_t sub_seq_id, size_t h) {
            quantize_block_by_channel<T, VALUE_DST_PREC>(v_src,
                                                         v_dst,
                                                         past_lens,
                                                         subsequence_begins,
                                                         block_indices,
                                                         block_indices_begins,
                                                         temp_buffer,
                                                         sub_seq_id,
                                                         h);
        });
    } else {
        parallel_for3d(B, L1, H, [&](size_t b, size_t m, size_t h) {
            auto slot = slot_mapping.ptr<int32_t>(b)[m];
            if (slot < 0) {
                return;
            }
            auto block_number = slot / block_size;
            auto block_offset = slot % block_size;
            quantize_block_by_dims<T, VALUE_DST_PREC>(v_src,
                                                      v_dst,
                                                      b,
                                                      h,
                                                      m,
                                                      block_number,
                                                      block_offset,
                                                      quant_param.value_group_size);
        });
    }
}

void attn_quantkv(const ov::intel_cpu::PlainTensor& k_src,
                  const ov::intel_cpu::PlainTensor& v_src,
                  float* temp_buffer,
                  const ov::intel_cpu::PlainTensor& k_dst,
                  const ov::intel_cpu::PlainTensor& v_dst,
                  const ov::intel_cpu::PlainTensor& k_scale_zp,
                  const ov::intel_cpu::PlainTensor& v_scale_zp,
                  const size_t L0,
                  const bool quant_k_by_channel,
                  const size_t k_group_size,
                  const size_t v_group_size) {
    if (k_src.get_precision() == ov::element::f32 && k_dst.get_precision() == ov::element::u8) {
        attn_quant_mt<float, uint8_t>(k_src,
                                      v_src,
                                      k_dst,
                                      v_dst,
                                      L0,
                                      temp_buffer,
                                      k_scale_zp,
                                      v_scale_zp,
                                      quant_k_by_channel,
                                      k_group_size,
                                      v_group_size);
    } else if (k_src.get_precision() == ov::element::bf16 && k_dst.get_precision() == ov::element::u8) {
        attn_quant_mt<ov::bfloat16, uint8_t>(k_src,
                                             v_src,
                                             k_dst,
                                             v_dst,
                                             L0,
                                             temp_buffer,
                                             k_scale_zp,
                                             v_scale_zp,
                                             quant_k_by_channel,
                                             k_group_size,
                                             v_group_size);
    } else if (k_src.get_precision() == ov::element::f16 && k_dst.get_precision() == ov::element::u8) {
        attn_quant_mt<ov::float16, uint8_t>(k_src,
                                            v_src,
                                            k_dst,
                                            v_dst,
                                            L0,
                                            temp_buffer,
                                            k_scale_zp,
                                            v_scale_zp,
                                            quant_k_by_channel,
                                            k_group_size,
                                            v_group_size);
    } else {
        OPENVINO_THROW("unsupport src type: ",
                       k_src.get_precision(),
                       ", dst type: ",
                       k_dst.get_precision(),
                       " in attn_quantkv");
    }
}

void paged_attn_quantkv(const ov::intel_cpu::PlainTensor& k_src,
                        const ov::intel_cpu::PlainTensor& v_src,
                        const ov::intel_cpu::PlainTensor& k_dst,
                        const ov::intel_cpu::PlainTensor& v_dst,
                        const ov::intel_cpu::PlainTensor& past_lens,
                        const ov::intel_cpu::PlainTensor& subsequence_begins,
                        const ov::intel_cpu::PlainTensor& block_indices,
                        const ov::intel_cpu::PlainTensor& block_indices_begins,
                        const ov::intel_cpu::PlainTensor& slot_mapping,
                        ov::intel_cpu::PlainTensor& temp_buffer,
                        const QuantizeParam& quant_param) {
    using function_type = void (*)(const ov::intel_cpu::PlainTensor&,
                                   const ov::intel_cpu::PlainTensor&,
                                   const ov::intel_cpu::PlainTensor&,
                                   const ov::intel_cpu::PlainTensor&,
                                   const ov::intel_cpu::PlainTensor&,
                                   const ov::intel_cpu::PlainTensor&,
                                   const ov::intel_cpu::PlainTensor&,
                                   const ov::intel_cpu::PlainTensor&,
                                   const ov::intel_cpu::PlainTensor&,
                                   ov::intel_cpu::PlainTensor&,
                                   const QuantizeParam&);
    static constexpr function_type funcs_fp32[] = {paged_attn_quant_mt<float, ov::element::u8, ov::element::u8>,
                                                   paged_attn_quant_mt<float, ov::element::u8, ov::element::u4>,
                                                   paged_attn_quant_mt<float, ov::element::u4, ov::element::u8>,
                                                   paged_attn_quant_mt<float, ov::element::u4, ov::element::u4>};
    static constexpr function_type funcs_bf16[] = {paged_attn_quant_mt<ov::bfloat16, ov::element::u8, ov::element::u8>,
                                                   paged_attn_quant_mt<ov::bfloat16, ov::element::u8, ov::element::u4>,
                                                   paged_attn_quant_mt<ov::bfloat16, ov::element::u4, ov::element::u8>,
                                                   paged_attn_quant_mt<ov::bfloat16, ov::element::u4, ov::element::u4>};
    static constexpr function_type funcs_f16[] = {paged_attn_quant_mt<ov::float16, ov::element::u8, ov::element::u8>,
                                                  paged_attn_quant_mt<ov::float16, ov::element::u8, ov::element::u4>,
                                                  paged_attn_quant_mt<ov::float16, ov::element::u4, ov::element::u8>,
                                                  paged_attn_quant_mt<ov::float16, ov::element::u4, ov::element::u4>};
    // sage attn
    static constexpr function_type saged_attn_fp32[] = {saged_attn_quant_mt<float, ov::element::i8, ov::element::u8>,
                                                        saged_attn_quant_mt<float, ov::element::u8, ov::element::u8>};
    static constexpr function_type saged_attn_bf16[] = {
        saged_attn_quant_mt<ov::bfloat16, ov::element::i8, ov::element::u8>,
        saged_attn_quant_mt<ov::bfloat16, ov::element::u8, ov::element::u8>};
    static constexpr function_type saged_attn_f16[] = {
        saged_attn_quant_mt<ov::float16, ov::element::i8, ov::element::u8>,
        saged_attn_quant_mt<ov::float16, ov::element::u8, ov::element::u8>};
    if (quant_param.is_sage_attn) {
        size_t dispatch = k_dst.get_precision() == ov::element::u8 ? 1 : 0;
        const auto sage_attn_call = [&]() -> function_type {
            switch (k_src.get_precision()) {
            case ov::element::f32:
                return saged_attn_fp32[dispatch];
            case ov::element::bf16:
                return saged_attn_bf16[dispatch];
            case ov::element::f16:
                return saged_attn_f16[dispatch];
            default:
                OPENVINO_THROW("Unsupported sage attention precision ", k_src.get_precision());
                return nullptr;
            }
        }();
        sage_attn_call(k_src,
                       v_src,
                       k_dst,
                       v_dst,
                       past_lens,
                       subsequence_begins,
                       block_indices,
                       block_indices_begins,
                       slot_mapping,
                       temp_buffer,
                       quant_param);
        return;
    }
    size_t dispatch = 0;
    if (k_dst.get_precision() == ov::element::u4) {
        dispatch |= 0x02;
    }
    if (v_dst.get_precision() == ov::element::u4) {
        dispatch |= 0x01;
    }
    if (k_src.get_precision() == ov::element::f32) {
        funcs_fp32[dispatch](k_src,
                             v_src,
                             k_dst,
                             v_dst,
                             past_lens,
                             subsequence_begins,
                             block_indices,
                             block_indices_begins,
                             slot_mapping,
                             temp_buffer,
                             quant_param);
    } else if (k_src.get_precision() == ov::element::bf16) {
        funcs_bf16[dispatch](k_src,
                             v_src,
                             k_dst,
                             v_dst,
                             past_lens,
                             subsequence_begins,
                             block_indices,
                             block_indices_begins,
                             slot_mapping,
                             temp_buffer,
                             quant_param);
    } else if (k_src.get_precision() == ov::element::f16) {
        funcs_f16[dispatch](k_src,
                            v_src,
                            k_dst,
                            v_dst,
                            past_lens,
                            subsequence_begins,
                            block_indices,
                            block_indices_begins,
                            slot_mapping,
                            temp_buffer,
                            quant_param);
    }
}

void attn_quant_u8(const float* src, uint8_t* dst, size_t n, float& scale, float& zp) {
    quant_u8(src, dst, n, scale, zp);
}
// u8 dequant needs scale + zp, params points to float[2]
void attn_dequant_u8(const uint8_t* src, float* dst, size_t n, float* params) {
    attn_dequant_kernel<float, ov::element::u8>(src, dst, n, params);
}

void attn_quant_by_channel_u8(const float* src,
                              uint8_t* dst,
                              size_t seq_dim,
                              size_t hidden_dims,
                              size_t src_stride,
                              size_t dst_stride,
                              float* scale,
                              float* zp) {
    quantize_by_channel<float, ov::element::u8>(src, dst, seq_dim, hidden_dims, src_stride, dst_stride, scale, zp);
}

void attn_dequant_by_channel_u8(const uint8_t* src,
                                float* dst,
                                size_t seq_dim,
                                size_t hidden_dims,
                                size_t src_stride,
                                size_t dst_stride,
                                float* scale,
                                float* zp) {
    attn_dequant_by_channel_kernel<float,
                                   ov::element::u8>(src, dst, seq_dim, hidden_dims, src_stride, dst_stride, scale, zp);
}

}  // namespace ov::Extensions::Cpu::XARCH
