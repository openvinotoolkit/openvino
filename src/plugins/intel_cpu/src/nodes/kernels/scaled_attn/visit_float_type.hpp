// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <type_traits>

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

/// @brief Calls f(std::integral_constant<ov::element::Type_t, PREC>{}) with the compressed cache precision `prec` (u4,
/// otherwise u8).
template <typename F>
void visit_quant_precision(ov::element::Type prec, const F& f) {
    if (prec == ov::element::u4) {
        f(std::integral_constant<ov::element::Type_t, ov::element::u4>{});
    } else {
        f(std::integral_constant<ov::element::Type_t, ov::element::u8>{});
    }
}

}  // namespace ov::Extensions::Cpu
