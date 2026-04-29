// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// Shared inner SIMD kernels consuming DecodePlan objects:
//   - decode_at: element-index -> plan.decoder.decode at correct bit offset
//   - codec_dot: generic QK dot product over dim elements
//   - codec_weighted_accum: generic V weighted accumulation into n_heads outputs
//
// These templates are header-only so each decoder-specific TU can
// instantiate them without cross-TU function calls.

#pragma once

#include <cstddef>
#include <cstdint>

#include "nodes/kernels/simd/simd_loop.hpp"

namespace ov::Extensions::Cpu::XARCH {

// Lightweight strided pointer — wraps (base, stride) for head-strided arrays.
// Usage: data[head][offset]
template <typename T>
struct StridedData {
    T* data;
    size_t stride;

    T* operator[](size_t i) const {
        return data + i * stride;
    }
};

// Decode elements at element index j from packed data using a DecodePlan.
// The plan carries both decoder (how bits become values) and params
// (resolved dequant context). j is NOT passed to the decoder.
template <typename Plan, simd::isa I>
auto decode_at(const uint8_t* data, int j, const Plan& plan, simd::active_lanes<I> a) {
    using Dec = std::decay_t<decltype(plan.decoder)>;
    constexpr int bits = Dec::bits;
    int b = j * bits;
    return plan.decoder.decode(data + b / 8, b % 8, plan.params, a);
}

// Attention QK dot product: sum(plan.decode(k[j]) * q[j]) for j in [0, dim).
// Returns raw dot product — caller applies outer scale.
// QT: query element type (float, float16, bfloat16). simd::load handles conversion.
// PlanFor: plan provider — plan_for(j, active_lanes<I>) returns a DecodePlan for position j.
// Uses simd_loop_reduce: 4x-unrolled SIMD main loop with vector accumulators,
// scalar tail, single final horizontal reduction.
template <typename QT, typename PlanFor>
float codec_dot(const uint8_t* k, const QT* q, int dim, PlanFor&& plan_for) {
    return simd::simd_loop_reduce<4>(
        dim,
        [&](int j, simd::f32& acc) {
            simd::active_lanes<simd::active_isa> a{};
            auto plan = plan_for(j, a);
            acc = simd::fmadd(simd::load<simd::f32>(q + j), decode_at(k, j, plan, a), acc);
        },
        [&](int j, float& tail) {
            simd::active_lanes<simd::isa::scalar> a{};
            auto plan = plan_for(j, a);
            using Vs = simd::f32_t<simd::isa::scalar>;
            tail += reduce(simd::load<Vs>(q + j) * decode_at(k, j, plan, a));
        });
}

// Attention V weighted accumulation: for each element j, decode v[j] once and
// accumulate into all n_heads output buffers with weight * outer_scale.
// PlanFor: plan provider — plan_for(j, active_lanes<I>) returns a DecodePlan for position j.
template <typename PlanFor>
void codec_weighted_accum(const uint8_t* v,
                          int dim,
                          PlanFor&& plan_for,
                          float outer_scale,
                          StridedData<const float> weights,
                          StridedData<float> accum,
                          int n_heads) {
    simd::simd_loop(dim, [&](int j, auto a) {
        auto plan = plan_for(j, a);
        auto v_dec = decode_at(v, j, plan, a);
        using V = std::decay_t<decltype(v_dec)>;
        for (int h = 0; h < n_heads; h++) {
            V vw(weights[h][0] * outer_scale);
            float* out = accum[h] + j;
            simd::store(simd::fmadd(vw, v_dec, simd::load<V>(out, a)), out, a);
        }
    });
}

}  // namespace ov::Extensions::Cpu::XARCH
