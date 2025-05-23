// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <vector>

#include "openvino/pass/matcher_pass.hpp"
#include "transformations_visibility.hpp"

namespace ov {
namespace pass {

class TRANSFORMATIONS_API ReshapeSinkingMatMul;

}  // namespace pass
}  // namespace ov

/**
 * @ingroup ov_transformation_common_api
 * @brief ReshapeSinkingMatMul transformation looks for MatMul followed by optional Add
 * surrounded with Reshape operations which are only needed to merge and unmerge dimensions
 * into MatMuls batch. In case of success upscales MatMul to work with multidimensional batch and updates
 * Reshape operators to make batch propagate through freely
 */

class ov::pass::ReshapeSinkingMatMul : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("ReshapeSinkingMatMul");
    ReshapeSinkingMatMul();
};
