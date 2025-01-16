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

class TRANSFORMATIONS_API TSSqueezeForward;
class TRANSFORMATIONS_API TSSqueezeBackward;

}  // namespace transpose_sinking
}  // namespace pass
}  // namespace ov

/**
 * @ingroup ov_transformation_common_api
 * @brief TSSqueezeForward transformation sinks Transpose through Reshape, Squeeze operations
 * in the forward direction.
 */
class ov::pass::transpose_sinking::TSSqueezeForward : public ov::pass::transpose_sinking::TSForwardBase {
public:
    OPENVINO_RTTI("ov::pass::TSSqueezeForward", "0");
    TSSqueezeForward();
};

/**
 * @ingroup ov_transformation_common_api
 * @brief TSSqueezeBackward transformation sinks Transpose through Reshape, Squeeze operations
 * in the backward direction.
 */
class ov::pass::transpose_sinking::TSSqueezeBackward : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("ov::pass::TSSqueezeBackward", "0");
    TSSqueezeBackward();
};
