// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <vector>

#include "openvino/pass/matcher_pass.hpp"
#include "transformations_visibility.hpp"

namespace ov {
namespace pass {

class TRANSFORMATIONS_API WeightsDequantizeToFakeQuantize;

}  // namespace pass
}  // namespace ov

/**
 * @ingroup ov_transformation_common_api
 * @brief WeightsDequantizeToFakeQuantize transformation replaces
 *      Constant (i8) -> Convert (to fp) -> Subtract (zp) -> Multiply (scale) ->
 *  with
 *      Constant (i8) -> Convert (to fp) -> FakeQuantize ->
 *  deducing levels and FakeQuantize limits according to actual values in the weights Constant
 */
class ov::pass::WeightsDequantizeToFakeQuantize : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("WeightsDequantizeToFakeQuantize", "0");
    WeightsDequantizeToFakeQuantize();
};
