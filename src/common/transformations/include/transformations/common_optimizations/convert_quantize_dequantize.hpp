// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <vector>

#include "openvino/core/type/element_type.hpp"
#include "openvino/pass/matcher_pass.hpp"
#include "transformations_visibility.hpp"

namespace ov {
namespace pass {

class TRANSFORMATIONS_API ConvertQuantizeDequantize;

}  // namespace pass
}  // namespace ov

/**
 * @ingroup ov_transformation_common_api
 * @brief ConvertQuantizeDequantize transformation replaces following graph:
 * FakeQuantize->Convert->Convert->Subtract->Multiply with a single FakeQuantize.
 * Restrictions:
 * - quantized data type must be i8, u8, i16, or u16
 * - 'levels' attribute to FakeQuantize must be equal to 256 or 65536
 * - (output_low, output_high) must match the quantized data type range
 * - 'zero_point' and 'scale' must be broadcastable to FakeQuantize's output
 * - supports mixed precision: quantizer and dequantizer can use different floating-point types (e.g., fp32/fp16)
 */

class ov::pass::ConvertQuantizeDequantize : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("ConvertQuantizeDequantize");
    ConvertQuantizeDequantize(const ov::element::TypeVector& supported_low_precisions = {ov::element::i8,
                                                                                         ov::element::u8,
                                                                                         ov::element::i16,
                                                                                         ov::element::u16},
                              const ov::element::TypeVector& supported_original_precisions = {ov::element::f32});
};
