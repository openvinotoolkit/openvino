// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision_transformations/subtract_multiply_to_multiply_add_transformation.hpp"

#include <memory>
#include <string>
#include <tuple>
#include <vector>

#include "ov_lpt_models/subtract_multiply_to_multiply_add.hpp"
#include "transformations/init_node_info.hpp"

namespace LayerTestsDefinitions {

std::string SubtractMultiplyToMultiplyAddTransformation::getTestCaseName(
    const testing::TestParamInfo<SubtractMultiplyToMultiplyAddTransformationParams>& obj) {
    std::string targetDevice;
    SubtractMultiplyToMultiplyAddTransformationTestValues testValues;
    std::tie(targetDevice, testValues) = obj.param;

    std::ostringstream result;
    result << targetDevice << "_" << testValues.inputShape << "_" << testValues.precision << "_" << testValues.fqOnData;
    return result.str();
}

void SubtractMultiplyToMultiplyAddTransformation::SetUp() {
    SubtractMultiplyToMultiplyAddTransformationTestValues testValues;
    std::tie(targetDevice, testValues) = this->GetParam();

    init_input_shapes(testValues.inputShape);

    function = ov::builder::subgraph::SubtractMultiplyToMultiplyAddFunction::getOriginal(testValues.inputShape,
                                                                                         testValues.precision,
                                                                                         testValues.fqOnData);
}

TEST_P(SubtractMultiplyToMultiplyAddTransformation, CompareWithRefImpl) {
    run();
};

}  // namespace LayerTestsDefinitions
