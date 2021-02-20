// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>
#include <memory>

#include <transformations_visibility.hpp>
#include <ngraph/pass/graph_rewrite.hpp>

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
class ngraph::pass::WeightsDequantizeToFakeQuantize: public ngraph::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    WeightsDequantizeToFakeQuantize();
};
