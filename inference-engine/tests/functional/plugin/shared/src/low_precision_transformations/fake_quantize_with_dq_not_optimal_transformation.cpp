// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision_transformations/fake_quantize_with_dq_not_optimal_transformation.hpp"

#include <memory>
#include <tuple>
#include <vector>
#include <string>
#include <ie_core.hpp>

#include <transformations/init_node_info.hpp>
#include "lpt_ngraph_functions/fake_quantize_and_convolution_function.hpp"

namespace LayerTestsDefinitions {

std::string FakeQuantizeWithNotOptimalTransformation::getTestCaseName(testing::TestParamInfo<FakeQuantizeTransformationParams> obj) {
    InferenceEngine::Precision netPrecision;
    InferenceEngine::SizeVector inputShapes;
    std::string targetDevice;
    ngraph::pass::low_precision::LayerTransformation::Params params;
    FakeQuantizeWithNotOptimalTransformationTestValues testValues;
    std::tie(netPrecision, inputShapes, targetDevice, params, testValues) = obj.param;

    std::ostringstream result;
    result << getTestCaseNameByParams(netPrecision, inputShapes, targetDevice, params) << "_" <<
        testValues.fqOnData << "_" <<
        testValues.dequantizationOnData << "_" <<
        testValues.fqOnWeights << "_" <<
        testValues.dequantizationOnWeights;
    return result.str();
}

void FakeQuantizeWithNotOptimalTransformation::SetUp() {
    InferenceEngine::SizeVector inputShape;
    InferenceEngine::Precision netPrecision;
    ngraph::pass::low_precision::LayerTransformation::Params params;
    FakeQuantizeWithNotOptimalTransformationTestValues testValues;
    std::tie(netPrecision, inputShape, targetDevice, params, testValues) = this->GetParam();

    function = ngraph::builder::subgraph::FakeQuantizeAndConvolutionFunction::get(
        FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision),
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

TEST_P(FakeQuantizeWithNotOptimalTransformation, CompareWithRefImpl) {
    Run();

    const auto actualType = getRuntimePrecision("output_original");
    EXPECT_EQ(actualType, "I8");
};

}  // namespace LayerTestsDefinitions
