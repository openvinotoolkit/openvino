// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision_transformations/concat_transformation.hpp"

#include <memory>
#include <tuple>
#include <vector>
#include <string>

#include "transformations/init_node_info.hpp"
#include "ov_lpt_models/concat.hpp"

namespace LayerTestsDefinitions {

std::string ConcatTransformation::getTestCaseName(const testing::TestParamInfo<ConcatTransformationParams>& obj) {
    auto [precision, inputShapes, targetDevice, testValues] = obj.param;

    std::ostringstream result;
    result << get_test_case_name_by_params(precision, inputShapes, targetDevice) <<
           testValues.fqOnData1 <<
        testValues.dequantization1 <<
        testValues.fqOnData2 <<
        testValues.dequantization2;
    return result.str();
}

void ConcatTransformation::SetUp() {
    auto [precision, inputShape, device, testValues] = this->GetParam();
    targetDevice = device;

    std::vector<ov::PartialShape> inputs;
    if (testValues.input_constant1 == nullptr) {
        inputs.push_back(inputShape);
    }
    if (testValues.input_constant2 == nullptr) {
        inputs.push_back(inputShape);
    }
    init_input_shapes(inputs);

    function = ov::builder::subgraph::ConcatFunction::getOriginal(
        precision,
        inputShape,
        testValues.input_constant1,
        testValues.fqOnData1,
        testValues.dequantization1,
        testValues.input_constant2,
        testValues.fqOnData2,
        testValues.dequantization2);
}

TEST_P(ConcatTransformation, CompareWithRefImpl) {
    run();
};

}  // namespace LayerTestsDefinitions
