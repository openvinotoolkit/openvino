// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision_transformations/subtract_multiply_to_multiply_add_transformation.hpp"

#include <memory>
#include <tuple>
#include <vector>
#include <string>
#include <ie_core.hpp>

#include <transformations/init_node_info.hpp>
#include "ngraph_functions/low_precision_transformations/subtract_multiply_to_multiply_add_function.hpp"

namespace LayerTestsDefinitions {

std::string SubtractMultiplyToMultiplyAddTransformation::getTestCaseName(testing::TestParamInfo<SubtractMultiplyToMultiplyAddTransformationParams> obj) {
    std::string targetDevice;
    SubtractMultiplyToMultiplyAddTransformationTestValues testValues;
    std::tie(targetDevice, testValues) = obj.param;

    std::ostringstream result;
    result <<
        targetDevice << "_" <<
        testValues.inputShape << "_" <<
        testValues.precision << "_" <<
        testValues.fqOnData;
    return result.str();
}

void SubtractMultiplyToMultiplyAddTransformation::SetUp() {
    SubtractMultiplyToMultiplyAddTransformationTestValues testValues;
    std::tie(targetDevice, testValues) = this->GetParam();

    function = ngraph::builder::subgraph::SubtractMultiplyToMultiplyAddFunction::getOriginal(
        testValues.inputShape,
        testValues.precision,
        testValues.fqOnData);

    validateNGraph();
}

void SubtractMultiplyToMultiplyAddTransformation::validateNGraph() {
    SubtractMultiplyToMultiplyAddTransformationTestValues testValues;
    std::tie(targetDevice, testValues) = this->GetParam();

    const ngraph::pass::low_precision::LayerTransformation::Params params = LayerTestsUtils::LayerTransformationParamsNGraphFactory::createParams();
    auto transformed = transformNGraph(params);

    ASSERT_EQ(1ul, transformed->get_output_size());
    std::shared_ptr<ngraph::Node> output = transformed->get_output_op(0);
    std::shared_ptr<ngraph::Node> scaleShift = output->get_input_node_shared_ptr(0);
    const std::string typeName = scaleShift->get_type_name();
    ASSERT_EQ("ScaleShiftIE", typeName);
}

TEST_P(SubtractMultiplyToMultiplyAddTransformation, CompareWithRefImpl) {
    Run();
};

}  // namespace LayerTestsDefinitions
