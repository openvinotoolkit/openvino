// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision_transformations/clamp_transformation.hpp"
#include <sstream>
#include <string>
#include <vector>
#include <ngraph/ngraph.hpp>

#include "lpt_ngraph_functions/clamp_function.hpp"

namespace LayerTestsDefinitions {

std::string ClampTransformation::getTestCaseName(testing::TestParamInfo<ClampTransformationParams> obj) {
    ngraph::element::Type netPrecision;
    ngraph::PartialShape inputShape;
    std::string targetDevice;
    ngraph::pass::low_precision::LayerTransformation::Params params;
    ClampTransformationParam param;;
    std::tie(netPrecision, inputShape, targetDevice, params, param) = obj.param;

    std::ostringstream result;
    result << getTestCaseNameByParams(netPrecision, inputShape, targetDevice, params) << "_" <<
        param.fakeQuantize << "_" <<
        "min=" << param.clampLowConst <<
        "max=" << param.clampHighConst;
    return result.str();
}

void ClampTransformation::SetUp() {
    ngraph::element::Type netPrecision;
    ngraph::PartialShape inputShape;
    ngraph::pass::low_precision::LayerTransformation::Params params;
    ClampTransformationParam param;
    std::tie(netPrecision, inputShape, targetDevice, params, param) = this->GetParam();

    function = ngraph::builder::subgraph::ClampFunction::getOriginal(
        netPrecision,
        inputShape,
        param.fakeQuantize,
        param.clampLowConst,
        param.clampHighConst);
}

TEST_P(ClampTransformation, CompareWithRefImpl) {
    Run();
};

} // namespace LayerTestsDefinitions
