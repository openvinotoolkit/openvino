// Copyright (C) 2022-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/op/fake_quantize.hpp"
#include "openvino/op/prelu.hpp"
#include "openvino/op/util/binary_elementwise_arithmetic.hpp"
#include "openvino/op/util/binary_elementwise_comparison.hpp"
#include "openvino/op/util/binary_elementwise_logical.hpp"
#include "openvino/pass/graph_rewrite.hpp"
#include "openvino/pass/pass.hpp"
#include "transformations/transpose_sinking/ts_base.hpp"
#include "transformations_visibility.hpp"

namespace ov {
namespace pass {
namespace transpose_sinking {

class TRANSFORMATIONS_API TSBinaryForward;
class TRANSFORMATIONS_API TSBinaryBackward;

}  // namespace transpose_sinking
}  // namespace pass
}  // namespace ov

/**
 * @ingroup ie_transformation_common_api
 * @brief TSBinaryForward transformation sinks Transpose through BinaryElementwiseArithmetic,
 * BinaryElementwiseComparison, BinaryElementwiseLogical and PRelu operations in the forward direction.
 */
class ov::pass::transpose_sinking::TSBinaryForward : public ov::pass::transpose_sinking::TSForwardBase {
public:
    OPENVINO_RTTI("ov::pass::TSBinaryForward", "0");
    TSBinaryForward();
};

/**
 * @ingroup ie_transformation_common_api
 * @brief TSBinaryBackward transformation sinks Transpose through BinaryElementwiseArithmetic,
 * BinaryElementwiseComparison, BinaryElementwiseLogical and PRelu operations in the backward direction.
 */
class ov::pass::transpose_sinking::TSBinaryBackward : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("ov::pass::TSBinaryBackward", "0");
    TSBinaryBackward();
};
