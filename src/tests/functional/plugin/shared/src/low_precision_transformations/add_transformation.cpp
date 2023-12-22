// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision_transformations/add_transformation.hpp"

#include <memory>
#include <tuple>
#include <vector>
#include <string>
#include <ie_core.hpp>

#include <transformations/init_node_info.hpp>
#include "ov_lpt_models/add.hpp"
#include "ov_models/subgraph_builders.hpp"

namespace LayerTestsDefinitions {

std::string AddTransformation::getTestCaseName(const testing::TestParamInfo< AddTransformationParams>& obj) {
    ngraph::element::Type netPrecision;
    ngraph::PartialShape inputShapes;
    std::string targetDevice;
    auto params = LayerTestsUtils::LayerTransformationParamsNGraphFactory::createParamsU8I8();
    AddTestValues param;
    std::tie(netPrecision, inputShapes, targetDevice, param) = obj.param;

    std::ostringstream result;
    result << getTestCaseNameByParams(netPrecision, inputShapes, targetDevice, params) <<
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

void AddTransformation::SetUp() {
    ngraph::element::Type precision;
    ngraph::PartialShape inputShape;
    AddTestValues param;
    std::tie(precision, inputShape, targetDevice, param) = this->GetParam();

    function = ngraph::builder::subgraph::AddFunction::getOriginal(
        precision, inputShape, param.broadcast,
        param.fakeQuantize1, param.fakeQuantize2);

    ov::pass::InitNodeInfo().run_on_model(function);
}

TEST_P(AddTransformation, CompareWithRefImpl) {
    Run();
};

}  // namespace LayerTestsDefinitions
