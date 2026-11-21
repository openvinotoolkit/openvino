// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/core/type/bfloat16.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/core/type/float16.hpp"

namespace ov::Extensions::Cpu {

/// @brief Calls f(T{}) with the C++ type of the floating-point precision `prec` (f32 unless bf16 / f16).
template <typename F>
void visit_float_type(ov::element::Type prec, const F& f) {
    if (prec == ov::element::bf16) {
        f(ov::bfloat16{});
    } else if (prec == ov::element::f16) {
        f(ov::float16{});
    } else {
        f(float{});
    }
}

}  // namespace ov::Extensions::Cpu
