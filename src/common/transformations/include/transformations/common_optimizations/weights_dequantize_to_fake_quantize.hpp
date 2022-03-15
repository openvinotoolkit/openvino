// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <ngraph/pass/graph_rewrite.hpp>
#include <transformations_visibility.hpp>
#include <vector>

namespace ngraph {
namespace pass {

class TRANSFORMATIONS_API WeightsDequantizeToFakeQuantize;

}  // namespace pass
}  // namespace ngraph

/**
 * @ingroup ie_transformation_common_api
 * @brief WeightsDequantizeToFakeQuantize transformation replaces
 *      Constant (i8) -> Convert (to fp) -> Subtract (zp) -> Multiply (scale) ->
 *  with
 *      Constant (i8) -> Convert (to fp) -> FakeQuantize ->
 *  deducing levels and FakeQuantize limits according to actual values in the weights Constant
 */
class ngraph::pass::WeightsDequantizeToFakeQuantize : public ngraph::pass::MatcherPass {
public:
    OPENVINO_RTTI("WeightsDequantizeToFakeQuantize", "0");
    WeightsDequantizeToFakeQuantize();
};
