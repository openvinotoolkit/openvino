// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>

#include "openvino/core/type/element_type.hpp"

namespace ov::Extensions::Cpu::XARCH {

void gated_residual_mix(const float* gate,
                        const float* rbar,
                        void* out,
                        ov::element::Type out_prc,
                        size_t data_size,
                        size_t branches,
                        float scale);

}  // namespace ov::Extensions::Cpu::XARCH
