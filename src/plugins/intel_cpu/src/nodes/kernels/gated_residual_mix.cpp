// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gated_residual_mix.hpp"

#include <cstddef>

#include "nodes/kernels/simd/simd_common.hpp"
#include "nodes/kernels/simd/simd_loop.hpp"
#include "openvino/core/except.hpp"
#include "openvino/core/type/bfloat16.hpp"
#include "openvino/core/type/element_type.hpp"

namespace ov::Extensions::Cpu::XARCH {

namespace {

template <typename DstT>
void mix_row(const float* gate, const float* rbar, DstT* out, size_t data_size, size_t branches, float scale) {
    simd::simd_loop(static_cast<int>(data_size), [&](int d, auto a) {
        // The tail runs at scalar width, so the vector type follows the lanes of this iteration.
        using V = simd::vec<float, decltype(a)::isa_tag::value>;
        V acc(0.0F);
        for (size_t c = 0; c < branches; c++) {
            const size_t offset = c * data_size + static_cast<size_t>(d);
            acc = fmadd(simd::load<V>(gate + offset, a), simd::load<V>(rbar + offset, a), acc);
        }
        simd::store(acc * V(scale), out + d, a);
    });
}

}  // namespace

void gated_residual_mix(const float* gate,
                        const float* rbar,
                        void* out,
                        ov::element::Type out_prc,
                        size_t data_size,
                        size_t branches,
                        float scale) {
    switch (out_prc) {
    case ov::element::f32:
        mix_row(gate, rbar, static_cast<float*>(out), data_size, branches, scale);
        break;
    case ov::element::bf16:
        mix_row(gate, rbar, static_cast<ov::bfloat16*>(out), data_size, branches, scale);
        break;
    default:
        OPENVINO_THROW("gated_residual_mix: unsupported output precision ", out_prc);
    }
}

}  // namespace ov::Extensions::Cpu::XARCH
