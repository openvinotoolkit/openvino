/*******************************************************************************
 * Copyright (c) 2018-2026 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *******************************************************************************/

// Shared helpers for the Gated Residual (hyper connection) CM kernels.

// A classic include guard, NOT `#pragma once`: the CM codegen inlines headers into
// each kernel separately, and the plugin then concatenates several kernels into one
// batched program. Only a textual guard survives that flattening and keeps the second
// copy from redefining these helpers.
#ifndef QSA_GR_COMMON_HPP
#define QSA_GR_COMMON_HPP

#include <cm/cm.h>
#include <cm/cmtl.h>

namespace qsa_gr {

constexpr float LOG2E = 1.4426950408889634f;

// Contiguous 16-element half load/store. `elem_off` must be a multiple of 2 so the
// underlying dword access stays aligned; all call sites use multiples of 16.
CM_INLINE vector<half, 16> load_h16(const half* p [[type("svmptr_t")]], uint elem_off) {
    vector<half, 16> v;
    v.format<int>() = cm_ptr_load<int, 8>((int*)p, elem_off * (uint)sizeof(half));
    return v;
}

CM_INLINE void store_h16(half* p [[type("svmptr_t")]], uint elem_off, vector<half, 16> v) {
    cm_ptr_store<int, 8>((int*)p, elem_off * (uint)sizeof(half), v.format<int>());
}

// Wide contiguous half load/store.
//
// Decode-side QSA kernels are DRAM-bandwidth bound, so the number of LSC messages
// matters as much as the number of bytes: a 128-element (256 B) load is ONE message
// while eight load_h16() calls are eight. The specializations mirror
// pa_kv_cache_update_ref.cm::load_kvcache so the same head sizes stay on a fast path
// (16/32/64/96/128/256; anything else falls back to a 16-element loop).
template <int N>
CM_INLINE void load_hv(vector_ref<half, N> dst, const half* p [[type("svmptr_t")]], uint elem_off) {
    static_assert(N % 16 == 0, "wide half load expects a multiple of 16 elements");
    const uint byte_off = elem_off * (uint)sizeof(half);
    if constexpr (N == 16 || N == 32 || N == 64 || N == 128) {
        dst.template format<int>() = cm_ptr_load<int, N / 2>((int*)p, byte_off);
    } else if constexpr (N == 96) {
        dst.template select<64, 1>(0).template format<int>() = cm_ptr_load<int, 32>((int*)p, byte_off);
        dst.template select<32, 1>(64).template format<int>() = cm_ptr_load<int, 16>((int*)p, byte_off + 32 * (uint)sizeof(int));
    } else if constexpr (N % 128 == 0) {
#pragma unroll
        for (int i = 0; i < N / 128; i++) {
            dst.template select<128, 1>(i * 128).template format<int>() = cm_ptr_load<int, 64>((int*)p, byte_off + i * 64 * (uint)sizeof(int));
        }
    } else {
#pragma unroll
        for (int i = 0; i < N / 16; i++) {
            dst.template select<16, 1>(i * 16).template format<int>() = cm_ptr_load<int, 8>((int*)p, byte_off + i * 8 * (uint)sizeof(int));
        }
    }
}

template <int N>
CM_INLINE void store_hv(half* p [[type("svmptr_t")]], uint elem_off, vector_ref<half, N> src) {
    static_assert(N % 16 == 0, "wide half store expects a multiple of 16 elements");
    const uint byte_off = elem_off * (uint)sizeof(half);
    if constexpr (N == 16 || N == 32 || N == 64 || N == 128) {
        cm_ptr_store<int, N / 2>((int*)p, byte_off, src.template format<int>());
    } else if constexpr (N == 96) {
        cm_ptr_store<int, 32>((int*)p, byte_off, src.template select<64, 1>(0).template format<int>());
        cm_ptr_store<int, 16>((int*)p, byte_off + 32 * (uint)sizeof(int), src.template select<32, 1>(64).template format<int>());
    } else if constexpr (N % 128 == 0) {
#pragma unroll
        for (int i = 0; i < N / 128; i++) {
            cm_ptr_store<int, 64>((int*)p, byte_off + i * 64 * (uint)sizeof(int), src.template select<128, 1>(i * 128).template format<int>());
        }
    } else {
#pragma unroll
        for (int i = 0; i < N / 16; i++) {
            cm_ptr_store<int, 8>((int*)p, byte_off + i * 8 * (uint)sizeof(int), src.template select<16, 1>(i * 16).template format<int>());
        }
    }
}

template <int N>
CM_INLINE vector<float, N> sigmoid(vector<float, N> x) {
    // cm_exp is exp2, hence the log2(e) rescaling of the exponent.
    return 1.0f / (1.0f + cm_exp(-x * LOG2E));
}

CM_INLINE float sigmoid_scalar(float x) {
    vector<float, 1> v = x;
    return sigmoid<1>(v)[0];
}

}  // namespace qsa_gr

#endif  // QSA_GR_COMMON_HPP
