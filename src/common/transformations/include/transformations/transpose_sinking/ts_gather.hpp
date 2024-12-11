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

class TRANSFORMATIONS_API TSGatherForward;
class TRANSFORMATIONS_API TSGatherBackward;

}  // namespace transpose_sinking
}  // namespace pass
}  // namespace ov

/**
 * @ingroup ov_transformation_common_api
 * @brief TSGatherForward transformation sinks Transpose through Gather operations
 * in the forward direction.
 */
class ov::pass::transpose_sinking::TSGatherForward : public ov::pass::transpose_sinking::TSForwardBase {
public:
    OPENVINO_RTTI("ov::pass::TSGatherForward", "0", ov::pass::transpose_sinking::TSForwardBase);
    TSGatherForward();
};

/**
 * @ingroup ov_transformation_common_api
 * @brief TSGatherBackward transformation sinks Transpose through Gather operation
 * in the backward direction.
 */
class ov::pass::transpose_sinking::TSGatherBackward : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("ov::pass::TSGatherBackward");
    TSGatherBackward();
};
