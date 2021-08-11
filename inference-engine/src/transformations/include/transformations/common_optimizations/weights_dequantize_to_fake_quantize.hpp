// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>
#include <memory>

#include <transformations_visibility.hpp>
#include <ngraph/pass/graph_rewrite.hpp>

namespace ov {
namespace pass {

class TRANSFORMATIONS_API WeightsDequantizeToFakeQuantize;

}  // namespace pass
}  // namespace ov

/**
 * @ingroup ie_transformation_common_api
 * @brief WeightsDequantizeToFakeQuantize transformation replaces
 *      Constant (i8) -> Convert (to fp) -> Subtract (zp) -> Multiply (scale) ->
 *  with
 *      Constant (i8) -> Convert (to fp) -> FakeQuantize ->
 *  deducing levels and FakeQuantize limits according to actual values in the weights Constant
 */
class ov::pass::WeightsDequantizeToFakeQuantize: public ov::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    WeightsDequantizeToFakeQuantize();
};
