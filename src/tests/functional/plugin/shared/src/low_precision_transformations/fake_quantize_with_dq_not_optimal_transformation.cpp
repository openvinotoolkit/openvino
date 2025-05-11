// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision_transformations/fake_quantize_with_dq_not_optimal_transformation.hpp"

#include <memory>
#include <tuple>
#include <vector>
#include <string>

#include "transformations/init_node_info.hpp"
#include "ov_lpt_models/fake_quantize_and_convolution.hpp"

namespace LayerTestsDefinitions {

std::string FakeQuantizeWithNotOptimalTransformation::getTestCaseName(const testing::TestParamInfo<FakeQuantizeTransformationParams>& obj) {
    ov::element::Type netPrecision;
    ov::PartialShape inputShapes;
    std::string targetDevice;
    ov::pass::low_precision::LayerTransformation::Params params;
    FakeQuantizeWithNotOptimalTransformationTestValues testValues;
    std::tie(netPrecision, inputShapes, targetDevice, params, testValues) = obj.param;

    std::ostringstream result;
    result << get_test_case_name_by_params(netPrecision, inputShapes, targetDevice, params) << "_" << testValues;
    return result.str();
}

void FakeQuantizeWithNotOptimalTransformation::SetUp() {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();
    ov::PartialShape inputShape;
    ov::element::Type netPrecision;
    ov::pass::low_precision::LayerTransformation::Params params;
    FakeQuantizeWithNotOptimalTransformationTestValues testValues;
    std::tie(netPrecision, inputShape, targetDevice, params, testValues) = this->GetParam();

    init_input_shapes(inputShape);

    function = ov::builder::subgraph::FakeQuantizeAndConvolutionFunction::get(
        netPrecision,
        inputShape,
        testValues.fqOnData,
        testValues.convertOnData,
        testValues.dequantizationOnData,
        testValues.constantOnWeights,
        testValues.fqOnWeights,
        testValues.convertOnWeights,
        testValues.dequantizationOnWeights,
        testValues.dequantizationAfter);
}

void FakeQuantizeWithNotOptimalTransformation::run() {
    LayerTransformation::run();

    const auto params = std::get<4>(GetParam());
    const auto actualType = get_runtime_precision_by_type("Convolution");
    EXPECT_EQ(actualType, params.expectedPrecision);
}

TEST_P(FakeQuantizeWithNotOptimalTransformation, CompareWithRefImpl) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();
    run();
};

}  // namespace LayerTestsDefinitions
