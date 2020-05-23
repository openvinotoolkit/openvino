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

class ngraph::pass::ConvertMulAddToScaleShiftOrPower: public ngraph::pass::GraphRewrite {
public:
    ConvertMulAddToScaleShiftOrPower() : GraphRewrite() {
        convert_mul_add_to_scaleshift_or_power();
    }

private:
    void convert_mul_add_to_scaleshift_or_power();
};

enum class CONVERSION_RESULT {
    SCALE_SHIFT,
    POWER,
    NONE
};

INFERENCE_ENGINE_API_CPP(CONVERSION_RESULT)
check_constant(const std::shared_ptr<ngraph::op::Constant> & constant, const ngraph::PartialShape & shape);
