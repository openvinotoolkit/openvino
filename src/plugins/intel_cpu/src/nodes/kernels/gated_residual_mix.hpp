// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>

#include "openvino/core/type/element_type.hpp"

namespace ov::Extensions::Cpu::XARCH {

// GatedResidual mix of one row: out[d] = scale * sum_c gate[c * data_size + d] * rbar[c * data_size + d]. gate and rbar
// are contiguous f32 rows of `branches * data_size` elements; out holds `data_size` elements of `out_prc` (f32, bf16 or
// f16), rounded once.
void gated_residual_mix(const float* gate,
                        const float* rbar,
                        void* out,
                        ov::element::Type out_prc,
                        size_t data_size,
                        size_t branches,
                        float scale);

}  // namespace ov::Extensions::Cpu::XARCH
