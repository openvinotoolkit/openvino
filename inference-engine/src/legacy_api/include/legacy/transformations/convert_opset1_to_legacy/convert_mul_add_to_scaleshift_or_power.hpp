// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>
#include <memory>

#include <ie_api.h>

#include <ngraph/pass/graph_rewrite.hpp>

namespace ngraph {
namespace pass {

class INFERENCE_ENGINE_API_CLASS(ConvertMulAddToScaleShiftOrPower);

}  // namespace pass
}  // namespace ngraph

class ngraph::pass::ConvertMulAddToScaleShiftOrPower: public ngraph::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    ConvertMulAddToScaleShiftOrPower();
};

enum class CONVERSION_RESULT {
    SCALE_SHIFT,
    POWER,
    NONE
};

/*
 * check_constant function checks how given constant performs elementwise operation with given input
 * CONVERSION_RESULT has several types:
 *      SCALE_SHIFT - constant applies only per-channel
 *      POWER - constant applies as single value
 *      NONE - default return value
 */

INFERENCE_ENGINE_API_CPP(CONVERSION_RESULT)
check_constant(const std::shared_ptr<ngraph::op::Constant> & constant, const ngraph::PartialShape & shape);
