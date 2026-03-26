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

class TRANSFORMATIONS_API HorizontalQDQFusion;

}  // namespace pass
}  // namespace ov

/**
 * @ingroup ov_transformation_common_api
 * @brief HorizontalQDQFusion detects multiple identical dequantization subgraphs
 * (Convert->Subtract->Multiply) sharing a common quantization Convert node and replaces
 * all duplicate DQ chains with a single shared one.
 *
 * The pattern matched is:
 * FakeQuantize -> Convert(to low precision, >1 consumer) -> Convert(to original precision)
 *   -> [Subtract(zero_point)] -> Multiply(scale)
 *
 * When the Convert(to low precision) node has multiple consumers that form identical
 * DQ subgraphs (same zero_point and scale values), the duplicates are replaced with
 * the original DQ subgraph matched by the pattern.
 */

class ov::pass::HorizontalQDQFusion : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("HorizontalQDQFusion");
    HorizontalQDQFusion(const ov::element::TypeVector& supported_low_precisions = {ov::element::i8,
                                                                                   ov::element::u8,
                                                                                   ov::element::i16,
                                                                                   ov::element::u16},
                        const ov::element::TypeVector& supported_original_precisions = {ov::element::f32});
};
