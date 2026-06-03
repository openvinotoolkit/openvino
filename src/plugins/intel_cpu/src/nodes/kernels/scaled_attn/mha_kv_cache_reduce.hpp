// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cassert>
#include <cstddef>

#include "codecs/turboq_rotation.hpp"
#include "cpu_parallel.hpp"
#include "nodes/kernels/simd/simd_loop.hpp"
#include "openvino/core/type/bfloat16.hpp"
#include "openvino/core/type/float16.hpp"
#include "utils/plain_tensor.hpp"

namespace ov::Extensions::Cpu::XARCH {

using ov::intel_cpu::CpuParallelPtr;
using ov::intel_cpu::PlainTensor;

// Sum per-thread accumulators: dst[i] = sum_t(src[i + t * stride]) for i in [0, dim).
// When T != float, converts f32 accumulator result to T on store.
// Uses simd_loop for automatic SIMD main loop + scalar tail handling.
template <typename T>
static void reduce_thread_accumulators(T* dst, const float* src, size_t stride, int dim, int nthr) {
    simd::simd_loop(dim, [&](int i, auto a) {
        using V = simd::vec<float, decltype(a)::isa_tag::value>;
        auto acc = simd::load<V>(src + i, a);
        const float* p = src + i + stride;
        for (int t = 1; t < nthr; t++) {
            acc = acc + simd::load<V>(p, a);
            p += stride;
        }
        simd::store(acc, dst + i, a);
    });
}

// Typed reduce: reduces thread accumulators + optional inverse rotation,
// writing the final result directly to typed output dst.
template <typename T>
static void
reduce_head(int dim, T* dst, float* src, int nthr, size_t stride, bool apply_inv_rotation, const float* signs) {
    if (!apply_inv_rotation) {
        reduce_thread_accumulators(dst, src, stride, dim, nthr);
    } else {
        assert(dim % 64 == 0 && "dim must be divisible by 64");
        assert(signs != nullptr && "WHT signs required for inverse rotation");
        reduce_thread_accumulators(src, src, stride, dim, nthr);
        turboq_wht_inverse(signs, src, dst, dim);
    }
}

template <typename T>
static void mha_reduce_typed(const PlainTensor& attn_score,
                             PlainTensor& output_emb,
                             bool has_out_transpose,
                             bool apply_inv_rotation,
                             size_t B,
                             size_t H,
                             size_t q_len,
                             size_t S,
                             int nthr,
                             const CpuParallelPtr& cpu_parallel,
                             size_t q_offset,
                             const float* signs) {
    const size_t thread_stride = attn_score.stride(0);

    auto reduce_body = [&](size_t b, size_t h, size_t m) {
        const size_t out_m = q_offset + m;
        auto* dst = has_out_transpose ? output_emb.ptr<T>(b, out_m, h * S) : output_emb.ptr<T>(b, h, out_m);
        auto* src0 = attn_score.ptr<float>(static_cast<size_t>(0), b, out_m, h);
        reduce_head(static_cast<int>(S), dst, src0, nthr, thread_stride, apply_inv_rotation, signs);
    };
    if (q_len == 1) {
        cpu_parallel->parallel_for2d(B, H, [&](size_t b, size_t h) {
            reduce_body(b, h, 0);
        });
    } else {
        cpu_parallel->parallel_for3d(B, H, q_len, reduce_body);
    }
}

static void mha_reduce(const PlainTensor& attn_score,
                       PlainTensor& output_emb,
                       bool has_out_transpose,
                       bool apply_inv_rotation,
                       size_t B,
                       size_t H,
                       size_t q_len,
                       size_t S,
                       int nthr,
                       const CpuParallelPtr& cpu_parallel,
                       const float* signs,
                       size_t q_offset = 0) {
    const auto out_prec = output_emb.get_precision();
    if (out_prec == ov::element::bf16) {
        mha_reduce_typed<ov::bfloat16>(attn_score,
                                       output_emb,
                                       has_out_transpose,
                                       apply_inv_rotation,
                                       B,
                                       H,
                                       q_len,
                                       S,
                                       nthr,
                                       cpu_parallel,
                                       q_offset,
                                       signs);
    } else if (out_prec == ov::element::f16) {
        mha_reduce_typed<ov::float16>(attn_score,
                                      output_emb,
                                      has_out_transpose,
                                      apply_inv_rotation,
                                      B,
                                      H,
                                      q_len,
                                      S,
                                      nthr,
                                      cpu_parallel,
                                      q_offset,
                                      signs);
    } else {
        mha_reduce_typed<float>(attn_score,
                                output_emb,
                                has_out_transpose,
                                apply_inv_rotation,
                                B,
                                H,
                                q_len,
                                S,
                                nthr,
                                cpu_parallel,
                                q_offset,
                                signs);
    }
}

}  // namespace ov::Extensions::Cpu::XARCH
