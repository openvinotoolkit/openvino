// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/matcher_pass.hpp"
#include "transformations_visibility.hpp"

namespace ov {
namespace pass {
namespace experimental {

class TRANSFORMATIONS_API ScaledShiftedClampFusion;

}  // namespace experimental
}  // namespace pass
}  // namespace ov

/**
 * @ingroup ov_transformation_common_api
 * @brief Fuses Clamp(Add(Multiply(x, scale_const), bias_const), lo, hi) into a single
 *        ov::op::experimental::ScaledShiftedClamp op. scale_const and bias_const must
 *        be scalar (shape_size == 1) Constants.
 * @note  Experimental. API subject to change. Plugins opt in by registering this pass.
 */
class ov::pass::experimental::ScaledShiftedClampFusion : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("ScaledShiftedClampFusion");
    ScaledShiftedClampFusion();
};
