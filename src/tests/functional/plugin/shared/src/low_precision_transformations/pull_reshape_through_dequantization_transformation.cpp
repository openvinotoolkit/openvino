// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision_transformations/pull_reshape_through_dequantization_transformation.hpp"

#include <memory>
#include <tuple>
#include <vector>
#include <string>


#include "common_test_utils/common_utils.hpp"
#include "ov_lpt_models/fake_quantize_and_convolution.hpp"

namespace LayerTestsDefinitions {

std::string PullReshapeThroughDequantizationTransformation::getTestCaseName(const testing::TestParamInfo<PullReshapeThroughDequantizationParams>& obj) {
    ov::element::Type netPrecision;
    ov::PartialShape inputShape;
    std::string targetDevice;
    ov::pass::low_precision::LayerTransformation::Params params;
    ov::Shape elementwiseConstantShapes;
    PullReshapeThroughDequantizationTestValues testValues;
    std::tie(netPrecision, inputShape, targetDevice, params, elementwiseConstantShapes, testValues) = obj.param;

    std::ostringstream result;
    result << get_test_case_name_by_params(netPrecision, inputShape, targetDevice, params) << "_" <<
           inputShape << "_" <<
        elementwiseConstantShapes << "_" <<
        testValues.precisionBeforeDequantization << "_" <<
        testValues.dequantizationOnActivations << "_" <<
        testValues.weights.outPrecision << "_" <<
        testValues.weights.values[0] << " _" <<
        testValues.dequantizationOnWeights;
    return result.str();
}

void PullReshapeThroughDequantizationTransformation::SetUp() {
    ov::element::Type netPrecision;
    ov::PartialShape inputShape;
    ov::pass::low_precision::LayerTransformation::Params params;
    ov::Shape elementwiseConstantShapes;
    PullReshapeThroughDequantizationTestValues testValues;
    std::tie(netPrecision, inputShape, targetDevice, params, elementwiseConstantShapes, testValues) = this->GetParam();

    init_input_shapes(inputShape);

    // to prevent test cases increasing let's parameterize test by dequantization shape and
    // initialize values here
    if (!testValues.dequantizationOnWeights.subtract.empty()) {
        testValues.dequantizationOnWeights.subtract.constantShape = elementwiseConstantShapes;
    }

    if (!testValues.dequantizationOnWeights.multiply.empty()) {
        testValues.dequantizationOnWeights.multiply.constantShape = elementwiseConstantShapes;
    }

    function = ov::builder::subgraph::FakeQuantizeAndConvolutionFunction::get(
        testValues.precisionBeforeDequantization,
        inputShape,
        testValues.fakeQuantizeOnData,
        {},
        testValues.dequantizationOnActivations,
        testValues.weights,
        {},
        {},
        testValues.dequantizationOnWeights,
        testValues.reshape1,
        testValues.multiply,
        testValues.transpose,
        testValues.reshape2,
        testValues.dequantizationAfter,
        "GroupConvolution");
}

void PullReshapeThroughDequantizationTransformation::run() {
    LayerTransformation::run();

    const auto params = std::get<5>(GetParam());
    const auto actualType = get_runtime_precision(params.operationName);
    EXPECT_EQ(actualType, params.expectedKernelType);
}

TEST_P(PullReshapeThroughDequantizationTransformation, CompareWithRefImpl) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();
    run();
};

}  // namespace LayerTestsDefinitions
