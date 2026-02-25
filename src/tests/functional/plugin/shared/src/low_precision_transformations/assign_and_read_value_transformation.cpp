// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision_transformations/assign_and_read_value_transformation.hpp"
#include <sstream>
#include <string>
#include <vector>

#include "ov_lpt_models/assign_and_read_value.hpp"

namespace LayerTestsDefinitions {

std::string AssignAndReadValueTransformation::getTestCaseName(const testing::TestParamInfo<AssignAndReadValueTransformationParams>& obj) {
    auto [netPrecision, inputShape, opset, device, param] = obj.param;
    std::ostringstream result;
    result << get_test_case_name_by_params(netPrecision, inputShape, device) << "_" <<
           param.fakeQuantize << "_" << opset;
    return result.str();
}

void AssignAndReadValueTransformation::SetUp() {
    auto [netPrecision, inputShape, opset, device, param] = this->GetParam();
    targetDevice = device;

    init_input_shapes(inputShape);

    function = ov::builder::subgraph::AssignAndReadValueFunction::getOriginal(
        netPrecision,
        inputShape,
        param.fakeQuantize,
        opset);
}

TEST_P(AssignAndReadValueTransformation, CompareWithRefImpl) {
    run();
};

} // namespace LayerTestsDefinitions
