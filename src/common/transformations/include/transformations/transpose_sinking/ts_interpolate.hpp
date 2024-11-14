// Copyright (C) 2022-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/matcher_pass.hpp"
#include "transformations/transpose_sinking/ts_base.hpp"
#include "transformations_visibility.hpp"

namespace ov {
namespace pass {
namespace transpose_sinking {

class TRANSFORMATIONS_API TSInterpolateForward;
class TRANSFORMATIONS_API TSInterpolateBackward;

}  // namespace transpose_sinking
}  // namespace pass
}  // namespace ov

/**
 * @ingroup ov_transformation_common_api
 * @brief TSInterpolateForward transformation sinks Transpose through Interpolate operation
 * in the forward direction.
 */
class ov::pass::transpose_sinking::TSInterpolateForward : public ov::pass::transpose_sinking::TSForwardBase {
public:
    OPENVINO_RTTI("ov::pass::TSInterpolateForward", "0");
    TSInterpolateForward();
};

/**
 * @ingroup ov_transformation_common_api
 * @brief TSInterpolateBackward transformation sinks Transpose through Interpolate operation
 * in the backward direction.
 */
class ov::pass::transpose_sinking::TSInterpolateBackward : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("ov::pass::TSInterpolateBackward", "0");
    TSInterpolateBackward();
};
