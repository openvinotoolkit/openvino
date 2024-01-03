// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision_transformations/pull_reshape_through_dequantization_transformation.hpp"

#include <memory>
#include <tuple>
#include <vector>
#include <string>

#include <ie_core.hpp>

#include "common_test_utils/common_utils.hpp"
#include "functional_test_utils/plugin_cache.hpp"
#include "shared_test_classes/base/layer_test_utils.hpp"
#include "functional_test_utils/blob_utils.hpp"
#include "ov_models/pass/convert_prc.hpp"
#include "ov_lpt_models/fake_quantize_and_convolution.hpp"

namespace LayerTestsDefinitions {

std::string PullReshapeThroughDequantizationTransformation::getTestCaseName(const testing::TestParamInfo<PullReshapeThroughDequantizationParams>& obj) {
    ngraph::element::Type netPrecision;
    ngraph::PartialShape inputShape;
    std::string targetDevice;
    ov::pass::low_precision::LayerTransformation::Params params;
    ngraph::Shape elementwiseConstantShapes;
    PullReshapeThroughDequantizationTestValues testValues;
    std::tie(netPrecision, inputShape, targetDevice, params, elementwiseConstantShapes, testValues) = obj.param;

    std::ostringstream result;
    result << getTestCaseNameByParams(netPrecision, inputShape, targetDevice, params) << "_" <<
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
    // threshold = 0.1f;

    ngraph::element::Type netPrecision;
    ngraph::PartialShape inputShape;
    ov::pass::low_precision::LayerTransformation::Params params;
    ngraph::Shape elementwiseConstantShapes;
    PullReshapeThroughDequantizationTestValues testValues;
    std::tie(netPrecision, inputShape, targetDevice, params, elementwiseConstantShapes, testValues) = this->GetParam();

    // to prevent test cases increasing let's parameterize test by dequantization shape and
    // initialize values here
    if (!testValues.dequantizationOnWeights.subtract.empty()) {
        testValues.dequantizationOnWeights.subtract.constantShape = elementwiseConstantShapes;
    }

    if (!testValues.dequantizationOnWeights.multiply.empty()) {
        testValues.dequantizationOnWeights.multiply.constantShape = elementwiseConstantShapes;
    }

    function = ngraph::builder::subgraph::FakeQuantizeAndConvolutionFunction::get(
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

void PullReshapeThroughDequantizationTransformation::Run() {
    LayerTestsCommon::Run();

    const auto params = std::get<5>(GetParam());
    const auto actualType = getRuntimePrecision(params.operationName);
    EXPECT_EQ(actualType, params.expectedKernelType);
}

TEST_P(PullReshapeThroughDequantizationTransformation, CompareWithRefImpl) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();
    Run();
};

}  // namespace LayerTestsDefinitions
