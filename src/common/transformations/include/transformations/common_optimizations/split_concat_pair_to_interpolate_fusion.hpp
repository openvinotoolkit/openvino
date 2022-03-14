// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <ngraph/ngraph.hpp>
#include <ngraph/pass/graph_rewrite.hpp>
#include <transformations_visibility.hpp>
#include <vector>

#include "ngraph/pattern/matcher.hpp"

namespace ngraph {
namespace pass {

class TRANSFORMATIONS_API SplitConcatPairToInterpolateFusion;

}  // namespace pass
}  // namespace ngraph

/**
 * @ingroup ie_transformation_common_api
 * @brief SplitConcatPairToInterpolateFusion transformation replaces group of
 * operations: Split -> Concat to Interpolate op.
 */
class ngraph::pass::SplitConcatPairToInterpolateFusion : public ngraph::pass::MatcherPass {
public:
    OPENVINO_RTTI("SplitConcatPairToInterpolateFusion", "0");
    SplitConcatPairToInterpolateFusion(bool use_shape_for_elimination = true);
};
