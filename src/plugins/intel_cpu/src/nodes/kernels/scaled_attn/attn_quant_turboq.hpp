// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>

#include "codecs.hpp"
#include "cpu_parallel.hpp"
#include "openvino/core/type/element_type.hpp"
#include "utils/plain_tensor.hpp"

namespace ov::Extensions::Cpu::XARCH {

// Quantize one KV head record using TurboQuant.
// Requires head_dim == 128, bits 3 or 4. Output size: turboq_head_bytes.
// src_precision: element type of src (f32, bf16, f16). bf16/f16 are converted during normalization.
void turboq_quantize_head(const void* src, void* dst, int head_dim, int bits, ov::element::Type src_precision);

// Fused Q·K dot product: unpack TBQ K record + codebook lookup + dot with rotated Q.
// Returns the dot product score without materializing a full float K buffer.
// K must be in rotated domain (no inverse rotation). Q must be pre-rotated.
float turboq_fused_qk_dot(const void* packed_k, const float* q_rotated, int head_dim, int bits);

// Fused V weighted accumulation: unpack + codebook lookup + scale + weighted add.
// accum[i] += weight * codebook[idx[i]] * norm * inv_sqrt_dim
// Operates directly on packed TBQ cache — no intermediate float buffer.
// num_heads outputs share the same unpacked V (for GQA).
void turboq_fused_v_accum(const void* packed_v,
                          const float* weights,
                          float* const* accum_ptrs,
                          int num_heads,
                          int head_dim,
                          int bits);

// Reduce nthr thread-local accumulators into dst using SIMD, then optionally apply R^T.
// src points to thread 0's accumulator; stride is the distance between threads' data.
// Reduce thread accumulators, optionally fuse QJL sign correction (S^T), then apply Q^T.
// sign_src=nullptr disables QJL sign reduction.
void turboq_reduce_head(int dim,
                        float* dst,
                        float* src,
                        int nthr,
                        size_t stride,
                        bool apply_inv_rotation,
                        float* sign_src,
                        size_t sign_stride);

// Returns packed byte size for one head record.
size_t turboq_head_bytes(int head_dim, int bits);
size_t turboq_head_bytes_qjl(int head_dim, int lm_bits);

// Returns packed byte size for a full KV row (all heads for one token).
size_t turboq_row_bytes(int num_kv_heads, int head_dim, int bits);

// ---------------------------------------------------------------------------
// PolarQuant API — integrated into the TBQ pipeline.
// ---------------------------------------------------------------------------

// Quantize one KV head record using PolarQuant recursive polar decomposition.
void polarq_quantize_head(const void* src, void* dst, int head_dim, int bits, ov::element::Type src_precision);

// Single-token polar QK dot / V accum (exported for unit tests).
float polarq_fused_qk_dot(const void* packed_k, const float* q, int head_dim, int bits);
void polarq_fused_v_accum(const void* packed_v,
                          const float* weights,
                          float* const* accum_ptrs,
                          int n_heads,
                          int head_dim,
                          int bits);

// Returns packed byte size for one PolarQuant head record.
size_t polarq_head_bytes(int head_dim, int bits);

// QJL quantize: (b-1)-bit Lloyd-Max + 1-bit sign correction.
void turboq_quantize_head_qjl(const void* src, void* dst, int head_dim, int lm_bits, ov::element::Type src_precision);

// QJL batch functions removed — QJL is experimental/disabled.

// TBQ/PolarQuant fused multi-head attention pipeline.
// k_bits: K codec bits (0 = K is raw float/f16/bf16 or u8, not TBQ/polar-packed).
// v_bits: V codec bits (0 = V is raw float/f16/bf16 or u8, >0 = TBQ/polar-packed).
// k_qjl/v_qjl: whether K/V use QJL (b-1 Lloyd-Max + 1-bit sign correction).
// k_polar/v_polar: whether K/V use PolarQuant (else TurboQuant).
// k_scale_zp: scale/zp tensor for u8 K (empty PlainTensor when K is f32/f16/bf16 or has codec).
// v_scale_zp: scale/zp tensor for u8 V (empty PlainTensor when V has codec).
// key_group_size: quantization group size for u8 K (ignored when K is not u8).
// value_group_size: quantization group size for u8 V (ignored when V is not u8).
// q_precision: element type of q_input (f32, bf16, or f16).
// kv_precision: element type of non-codec K/V cache (f32, f16, bf16). Ignored when K/V has codec.
void mha_turboq(ov::intel_cpu::PlainTensor& q_input,
                const ov::intel_cpu::PlainTensor& key_cache,
                const ov::intel_cpu::PlainTensor& packed_value,
                const ov::intel_cpu::PlainTensor& alibi_mask,
                const ov::intel_cpu::PlainTensor& attention_mask,
                const ov::intel_cpu::PlainTensor& beams,
                ov::intel_cpu::PlainTensor& output_emb,
                ov::intel_cpu::PlainTensor& buf_attn_w,
                ov::intel_cpu::PlainTensor& buf_attn_score,
                bool has_out_transpose,
                float d_scale,
                ov::Extensions::Cpu::CacheCodec k_codec,
                ov::Extensions::Cpu::CacheCodec v_codec,
                bool v_rotation_fused,
                bool auto_causal,
                const ov::intel_cpu::PlainTensor& sink_input,
                const ov::intel_cpu::CpuParallelPtr& cpu_parallel,
                const ov::intel_cpu::PlainTensor& k_scale_zp,
                size_t key_group_size,
                const ov::intel_cpu::PlainTensor& v_scale_zp,
                size_t value_group_size,
                ov::element::Type q_precision,
                size_t value_head_dim);

}  // namespace ov::Extensions::Cpu::XARCH
