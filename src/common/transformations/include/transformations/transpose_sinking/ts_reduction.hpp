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

class TRANSFORMATIONS_API TSReductionForward;
class TRANSFORMATIONS_API TSReductionBackward;

}  // namespace transpose_sinking
}  // namespace pass
}  // namespace ov

/**
 * @ingroup ov_transformation_common_api
 * @brief TSReductionForward transformation sinks Transpose through Reduce operations
 * in the forward direction.
 */
class ov::pass::transpose_sinking::TSReductionForward : public ov::pass::transpose_sinking::TSForwardBase {
public:
    OPENVINO_RTTI("ov::pass::TSReductionForward", "0");
    TSReductionForward();
};

/**
 * @ingroup ov_transformation_common_api
 * @brief TSReductionBackward transformation sinks Transpose through Reduce operations
 * in the backward direction.
 */
class ov::pass::transpose_sinking::TSReductionBackward : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("ov::pass::TSReductionBackward", "0");
    TSReductionBackward();
};