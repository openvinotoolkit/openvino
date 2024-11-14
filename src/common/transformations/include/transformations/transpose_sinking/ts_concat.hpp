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

class TRANSFORMATIONS_API TSConcatForward;
class TRANSFORMATIONS_API TSConcatBackward;

}  // namespace transpose_sinking
}  // namespace pass
}  // namespace ov

/**
 * @ingroup ov_transformation_common_api
 * @brief TSConcatForward transformation sinks Transpose through Concat operation
 * in the forward direction.
 */
class ov::pass::transpose_sinking::TSConcatForward : public ov::pass::transpose_sinking::TSForwardBase {
public:
    OPENVINO_RTTI("ov::pass::TSConcatForward", "0");
    TSConcatForward();
};

/**
 * @ingroup ov_transformation_common_api
 * @brief TSConcatBackward transformation sinks Transpose through Concat operation
 * in the backward direction.
 */
class ov::pass::transpose_sinking::TSConcatBackward : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("ov::pass::TSConcatBackward", "0");
    TSConcatBackward();
};
