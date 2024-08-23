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

class TRANSFORMATIONS_API SplitConcatPairToInterpolateFusion;

}  // namespace pass
}  // namespace ov

/**
 * @ingroup ov_transformation_common_api
 * @brief SplitConcatPairToInterpolateFusion transformation replaces group of
 * operations: Split -> Concat to Interpolate op.
 */
class ov::pass::SplitConcatPairToInterpolateFusion : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("SplitConcatPairToInterpolateFusion", "0");
    SplitConcatPairToInterpolateFusion(bool use_shape_for_elimination = true);
};
