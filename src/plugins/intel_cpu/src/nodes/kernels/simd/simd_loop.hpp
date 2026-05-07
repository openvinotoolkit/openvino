// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// simd_loop: unified SIMD iteration with automatic tail handling.
//
// The loop body is written once, generic over ISA. The loop driver
// instantiates it at the active ISA for the main loop and at scalar
// ISA for the tail (x86/NEON) or with a predicate (SVE/RVV, future).
//
// See simd_loop.md (RFC) for design rationale.

#pragma once

#include <type_traits>
#include <utility>

#include "simd.hpp"

namespace ov::Extensions::Cpu::XARCH::simd {

// active_lanes<I> is defined in simd_common.hpp — available here via simd.hpp.

// ---------------------------------------------------------------------------
// Active-aware load/store: thin wrappers that delegate to existing per-ISA
// free functions. The active_lanes parameter selects the ISA at compile time.
// Future: SVE/RVV overloads will use the predicate/vl from active_lanes.
// ---------------------------------------------------------------------------

// Load: any source type (float, bf16, f16, u8) → vec<float, I>.
template <typename V, isa I, typename SrcT>
inline V load(const SrcT* ptr, active_lanes<I>) {
    return load(ptr, static_cast<V*>(nullptr));
}

// Store: vec<float, I> → any destination type (float, bf16, f16).
template <typename V, typename DstT, isa I>
inline void store(V v, DstT* ptr, active_lanes<I>) {
    store(v, ptr);
}

// Reduce: horizontal sum of active lanes → scalar float.
// For scalar ISA, just returns the value. For SIMD, reduces all lanes.
// Future: SVE/RVV will reduce only predicated/active lanes.
template <typename V, isa I>
inline float reduce(V v, active_lanes<I>) {
    return reduce(v);
}

// ---------------------------------------------------------------------------
// for_each_chunk<I>: ISA-specific loop driver.
//
// x86 (AVX2/AVX512) and NEON: SIMD main loop + scalar tail.
// Scalar: element-by-element.
// Future SVE: single predicated loop.
// Future RVV: single VL-adaptive loop.
//
// Body signature: body(int offset, active_lanes<I> a)
// Body is called with active_lanes<I> for full-width iterations and
// active_lanes<isa::scalar> for tail elements.
// ---------------------------------------------------------------------------

template <isa I, typename Body>
inline void for_each_chunk(int n, Body&& body) {
    if constexpr (I == isa::scalar) {
        for (int j = 0; j < n; j++) {
            body(j, active_lanes<isa::scalar>{});
        }
    } else {
        constexpr int W = vec<float, I>::width;
        int j = 0;
        for (; j + W - 1 < n; j += W) {
            body(j, active_lanes<I>{});
        }
        for (; j < n; j++) {
            body(j, active_lanes<isa::scalar>{});
        }
    }
}

// ---------------------------------------------------------------------------
// simd_loop: public frontend — dispatches to the active ISA loop driver.
// ---------------------------------------------------------------------------

template <typename Body>
inline void simd_loop(int n, Body&& body) {
    for_each_chunk<active_isa>(n, std::forward<Body>(body));
}

// ---------------------------------------------------------------------------
// Compile-time unroll helper.
// Calls fn(integral_constant<int, 0>{}), fn(integral_constant<int, 1>{}), ...
// ---------------------------------------------------------------------------

template <int N, typename Fn, int... Is>
inline void unroll_impl(Fn&& fn, std::integer_sequence<int, Is...>) {
    (fn(std::integral_constant<int, Is>{}), ...);
}

template <int N, typename Fn>
inline void unroll(Fn&& fn) {
    unroll_impl<N>(std::forward<Fn>(fn), std::make_integer_sequence<int, N>{});
}

// ---------------------------------------------------------------------------
// simd_loop_reduce: reduction-oriented SIMD loop with explicit vector
// accumulators, compile-time unrolling, and scalar tail.
//
// Maintains Unroll independent vector accumulators for FMA latency hiding.
// main_body(int offset, f32& acc): called per W-wide SIMD chunk.
//   Driver handles unrolling — body sees one chunk at a time.
// tail_body(int offset, float& tail): called per scalar tail element.
// Returns: reduce(acc[0] + acc[1] + ... + acc[Unroll-1]) + tail.
//
// For hot reduction kernels (dot product, norm, score accumulation) where
// per-iteration reduce in simd_loop would be too expensive.
// ---------------------------------------------------------------------------

template <int Unroll = 4, typename MainBody, typename TailBody>
inline float simd_loop_reduce(int n, MainBody&& main_body, TailBody&& tail_body) {
    constexpr int W = f32::width;

    f32 acc[Unroll]{};
    float tail_acc = 0.0F;
    int j = 0;

    // Unrolled main loop: Unroll × W elements per outer iteration.
    for (; j + Unroll * W - 1 < n; j += Unroll * W) {
        unroll<Unroll>([&](auto U) {
            main_body(j + int(U) * W, acc[int(U)]);
        });
    }
    // Single-width cleanup: remaining full W-wide chunks.
    for (; j + W - 1 < n; j += W) {
        main_body(j, acc[0]);
    }
    // Scalar tail.
    for (; j < n; j++) {
        tail_body(j, tail_acc);
    }

    // Final reduction: combine all vector accumulators + scalar tail.
    f32 combined = acc[0];
    for (int u = 1; u < Unroll; u++) {
        combined = combined + acc[u];
    }
    return reduce(combined) + tail_acc;
}

}  // namespace ov::Extensions::Cpu::XARCH::simd
