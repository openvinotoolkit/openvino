// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ie_api.h>

#include <memory>
#include <ngraph/pass/graph_rewrite.hpp>
#include <vector>

namespace ngraph {
namespace pass {

class ConvertMulAddToScaleShiftOrPower;

}  // namespace pass
}  // namespace ngraph

class ngraph::pass::ConvertMulAddToScaleShiftOrPower : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("ConvertMulAddToScaleShiftOrPower", "0");
    ConvertMulAddToScaleShiftOrPower();
};

enum class CONVERSION_RESULT { SCALE_SHIFT, POWER, NONE };

/*
 * check_constant function checks how given constant performs elementwise operation with given input
 * CONVERSION_RESULT has several types:
 *      SCALE_SHIFT - constant applies only per-channel
 *      POWER - constant applies as single value
 *      NONE - default return value
 */

CONVERSION_RESULT
check_constant(const std::shared_ptr<ngraph::op::Constant>& constant, const ngraph::PartialShape& shape);
