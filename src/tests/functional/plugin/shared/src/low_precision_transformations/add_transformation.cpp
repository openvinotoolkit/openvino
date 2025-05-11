// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision_transformations/add_transformation.hpp"

#include <memory>
#include <tuple>
#include <vector>
#include <string>

#include "transformations/init_node_info.hpp"
#include "ov_lpt_models/add.hpp"

namespace LayerTestsDefinitions {

std::string AddTransformation::getTestCaseName(const testing::TestParamInfo< AddTransformationParams>& obj) {
    ov::element::Type netPrecision;
    ov::PartialShape inputShapes;
    std::string targetDevice;
    auto params = LayerTestsUtils::LayerTransformationParamsNGraphFactory::createParamsU8I8();
    AddTestValues param;
    std::tie(netPrecision, inputShapes, targetDevice, param) = obj.param;

    std::ostringstream result;
    result << get_test_case_name_by_params(netPrecision, inputShapes, targetDevice, params) <<
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
    ov::element::Type precision;
    ov::PartialShape inputShape;
    AddTestValues param;
    std::tie(precision, inputShape, targetDevice, param) = this->GetParam();

    ov::PartialShape inputShape2 = inputShape;
    if (param.broadcast) {
        inputShape2[2] = 1;
        inputShape2[3] = 1;
    }
    init_input_shapes({ inputShape, inputShape2 });

    function = ov::builder::subgraph::AddFunction::getOriginal(
        precision, inputShape, param.broadcast,
        param.fakeQuantize1, param.fakeQuantize2);

    ov::pass::InitNodeInfo().run_on_model(function);
}

TEST_P(AddTransformation, CompareWithRefImpl) {
    run();
};

}  // namespace LayerTestsDefinitions
