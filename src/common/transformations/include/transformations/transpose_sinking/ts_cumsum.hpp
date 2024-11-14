// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/matcher_pass.hpp"
#include "transformations/transpose_sinking/ts_base.hpp"
#include "transformations_visibility.hpp"

namespace ov {
namespace pass {
namespace transpose_sinking {

class TRANSFORMATIONS_API TSCumSumForward;
class TRANSFORMATIONS_API TSCumSumBackward;

}  // namespace transpose_sinking
}  // namespace pass
}  // namespace ov

/**
 * @ingroup ov_transformation_common_api
 * @brief TSCumSumForward transformation sinks Transpose through CumSum in the forward direction.
 */
class ov::pass::transpose_sinking::TSCumSumForward : public ov::pass::transpose_sinking::TSForwardBase {
public:
    OPENVINO_RTTI("ov::pass::TSBinaryForward", "0");
    TSCumSumForward();
};

/**
 * @ingroup ov_transformation_common_api
 * @brief TSCumSumBackward transformation sinks Transpose through CumSum in the backward direction.
 */
class ov::pass::transpose_sinking::TSCumSumBackward : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("ov::pass::TSBinaryBackward", "0");
    TSCumSumBackward();
};
