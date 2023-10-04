// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision_transformations/assign_and_read_value_transformation.hpp"
#include <sstream>
#include <string>
#include <vector>
#include <ngraph/ngraph.hpp>

#include "ov_lpt_models/assign_and_read_value.hpp"

namespace LayerTestsDefinitions {

std::string AssignAndReadValueTransformation::getTestCaseName(const testing::TestParamInfo<AssignAndReadValueTransformationParams>& obj) {
    ngraph::element::Type netPrecision;
    ngraph::PartialShape inputShape;
    size_t opset;
    std::string targetDevice;
    ov::pass::low_precision::LayerTransformation::Params params;
    AssignAndReadValueTransformationParam param;;
    std::tie(netPrecision, inputShape, opset, targetDevice, params, param) = obj.param;

    std::ostringstream result;
    result << getTestCaseNameByParams(netPrecision, inputShape, targetDevice, params) << "_" <<
        param.fakeQuantize << "_" << opset;
    return result.str();
}

void AssignAndReadValueTransformation::SetUp() {
    ngraph::element::Type netPrecision;
    ngraph::PartialShape inputShape;
    size_t opset;
    ov::pass::low_precision::LayerTransformation::Params params;
    AssignAndReadValueTransformationParam param;
    std::tie(netPrecision, inputShape, opset, targetDevice, params, param) = this->GetParam();

    function = ngraph::builder::subgraph::AssignAndReadValueFunction::getOriginal(
        netPrecision,
        inputShape,
        param.fakeQuantize,
        opset);
}

TEST_P(AssignAndReadValueTransformation, CompareWithRefImpl) {
    Run();
};

} // namespace LayerTestsDefinitions
