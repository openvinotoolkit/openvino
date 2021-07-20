// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision_transformations/multiply_transformation.hpp"

#include <memory>
#include <tuple>
#include <vector>
#include <string>
#include <ie_core.hpp>
#include <transformations/init_node_info.hpp>

#include "lpt_ngraph_functions/multiply_function.hpp"
#include "ngraph_functions/subgraph_builders.hpp"


namespace LayerTestsDefinitions {

std::string MultiplyTransformation::getTestCaseName(testing::TestParamInfo<MultiplyTransformationParams> obj) {
    ngraph::element::Type precision;
    ngraph::PartialShape inputShapes;
    std::string targetDevice;
    auto params = LayerTestsUtils::LayerTransformationParamsNGraphFactory::createParamsU8I8();
    MultiplyTestValues param;
    std::tie(precision, inputShapes, targetDevice, param) = obj.param;

    std::ostringstream result;
    result << getTestCaseNameByParams(precision, inputShapes, targetDevice, params) <<
        (param.broadcast ? "_broadcast" : "");
    for (const auto& elem : param.precisionOnActivations) {
        result << "_" << elem << "_";
    }
    result << "expected_precisions_";
    for (const auto& elem : param.expectedPrecisions) {
        result << "_" << elem << "_";
    }

    if (!param.fakeQuantize1.empty()) {
        result << "_on_branch1_" <<
            param.fakeQuantize1.inputLowValues[0] << "_" <<
            param.fakeQuantize1.inputHighValues[0] << "_" <<
            param.fakeQuantize1.outputLowValues[0] << "_" <<
            param.fakeQuantize1.outputHighValues[0];
    }
    if (!param.fakeQuantize2.empty()) {
        result << "_on_branch2_" <<
            param.fakeQuantize2.inputLowValues[0] << "_" <<
            param.fakeQuantize2.inputHighValues[0] << "_" <<
            param.fakeQuantize2.outputLowValues[0] << "_" <<
            param.fakeQuantize2.outputHighValues[0];
    }
    return result.str();
}

void MultiplyTransformation::SetUp() {
    ngraph::element::Type precision;
    ngraph::PartialShape inputShape;
    MultiplyTestValues param;
    std::tie(precision, inputShape, targetDevice, param) = this->GetParam();

    function = ngraph::builder::subgraph::MultiplyFunction::getOriginal(
        precision,
        inputShape,
        param.broadcast,
        param.fakeQuantize1,
        param.fakeQuantize2);

    ngraph::pass::InitNodeInfo().run_on_function(function);
}

TEST_P(MultiplyTransformation, CompareWithRefImpl) {
    Run();
};

}  // namespace LayerTestsDefinitions
