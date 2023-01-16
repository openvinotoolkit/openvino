// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <openvino/pass/graph_rewrite.hpp>
#include <openvino/pass/pattern/matcher.hpp>
#include <transformations_visibility.hpp>
#include <vector>

namespace ov {
namespace pass {

class TRANSFORMATIONS_API SplitConcatPairToInterpolateFusion;

}  // namespace pass
}  // namespace ov

/**
 * @ingroup ie_transformation_common_api
 * @brief SplitConcatPairToInterpolateFusion transformation replaces group of
 * operations: Split -> Concat to Interpolate op.
 */
class ov::pass::SplitConcatPairToInterpolateFusion : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("SplitConcatPairToInterpolateFusion", "0");
    SplitConcatPairToInterpolateFusion(bool use_shape_for_elimination = true);
};

namespace ngraph {
namespace pass {
using ov::pass::SplitConcatPairToInterpolateFusion;
}  // namespace pass
}  // namespace ngraph
