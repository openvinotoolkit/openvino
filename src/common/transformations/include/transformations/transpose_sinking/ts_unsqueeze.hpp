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

class TRANSFORMATIONS_API TSUnsqueezeForward;
class TRANSFORMATIONS_API TSUnsqueezeBackward;

}  // namespace transpose_sinking
}  // namespace pass
}  // namespace ov

/**
 * @ingroup ov_transformation_common_api
 * @brief TSUnsqueezeForward transformation sinks Transpose through Unsqueeze, Reshape operations
 * in the forward direction.
 */
class ov::pass::transpose_sinking::TSUnsqueezeForward : public ov::pass::transpose_sinking::TSForwardBase {
public:
    OPENVINO_RTTI("ov::pass::TSUnsqueezeForward", "0", ov::pass::transpose_sinking::TSForwardBase);
    TSUnsqueezeForward();
};

/**
 * @ingroup ov_transformation_common_api
 * @brief TSUnsqueezeBackward transformation sinks Transpose through Unsqueeze, Reshape operations
 * in the backward direction.
 */
class ov::pass::transpose_sinking::TSUnsqueezeBackward : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("ov::pass::TSUnsqueezeBackward");
    TSUnsqueezeBackward();
};
