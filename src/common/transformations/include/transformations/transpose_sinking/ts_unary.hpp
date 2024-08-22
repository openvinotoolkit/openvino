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

class TRANSFORMATIONS_API TSUnaryForward;
class TRANSFORMATIONS_API TSUnaryBackward;

}  // namespace transpose_sinking
}  // namespace pass
}  // namespace ov

/**
 * @ingroup ov_transformation_common_api
 * @brief TSUnaryForward transformation sinks Transpose through UnaryElementwiseArithmetic, Clamp, Elu,
 * SoftPlus, LogicalNot, Convert, IsInf, IsNaN, IsFinite operations in the forward direction.
 */
class ov::pass::transpose_sinking::TSUnaryForward : public ov::pass::transpose_sinking::TSForwardBase {
public:
    OPENVINO_RTTI("TSUnaryForward", "0");
    TSUnaryForward();
};

/**
 * @ingroup ov_transformation_common_api
 * @brief TSUnaryBackward transformation sinks Transpose through UnaryElementwiseArithmetic, Clamp, Elu,
 * SoftPlus, LogicalNot, Convert, IsInf, IsNaN, IsFinite in the backward direction.
 */
class ov::pass::transpose_sinking::TSUnaryBackward : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("TSUnaryBackwardMultiConsumers", "0");
    TSUnaryBackward();
};
