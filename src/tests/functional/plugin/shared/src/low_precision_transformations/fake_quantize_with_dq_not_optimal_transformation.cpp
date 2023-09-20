// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision_transformations/fake_quantize_with_dq_not_optimal_transformation.hpp"

#include <memory>
#include <tuple>
#include <vector>
#include <string>
#include <ie_core.hpp>

#include <transformations/init_node_info.hpp>
#include "ov_lpt_models/fake_quantize_and_convolution.hpp"

namespace LayerTestsDefinitions {

std::string FakeQuantizeWithNotOptimalTransformation::getTestCaseName(const testing::TestParamInfo<FakeQuantizeTransformationParams>& obj) {
    ngraph::element::Type netPrecision;
    ngraph::PartialShape inputShapes;
    std::string targetDevice;
    ov::pass::low_precision::LayerTransformation::Params params;
    FakeQuantizeWithNotOptimalTransformationTestValues testValues;
    std::tie(netPrecision, inputShapes, targetDevice, params, testValues) = obj.param;

    std::ostringstream result;
    result << getTestCaseNameByParams(netPrecision, inputShapes, targetDevice, params) << "_" << testValues;
    return result.str();
}

void FakeQuantizeWithNotOptimalTransformation::SetUp() {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();
    ngraph::PartialShape inputShape;
    ngraph::element::Type netPrecision;
    ov::pass::low_precision::LayerTransformation::Params params;
    FakeQuantizeWithNotOptimalTransformationTestValues testValues;
    std::tie(netPrecision, inputShape, targetDevice, params, testValues) = this->GetParam();

    function = ngraph::builder::subgraph::FakeQuantizeAndConvolutionFunction::get(
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

void FakeQuantizeWithNotOptimalTransformation::Run() {
    LayerTestsCommon::Run();

    const auto params = std::get<4>(GetParam());
    const auto actualType = getRuntimePrecisionByType("Convolution");
    EXPECT_EQ(actualType, params.expectedPrecision);
}

TEST_P(FakeQuantizeWithNotOptimalTransformation, CompareWithRefImpl) {
    Run();
};

}  // namespace LayerTestsDefinitions
