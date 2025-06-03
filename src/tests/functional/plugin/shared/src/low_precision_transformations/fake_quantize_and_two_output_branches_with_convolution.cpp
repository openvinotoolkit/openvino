// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision_transformations/fake_quantize_and_two_output_branches_with_convolution.hpp"

#include <memory>
#include <tuple>
#include <vector>
#include <string>


#include "common_test_utils/common_utils.hpp"

namespace LayerTestsDefinitions {

std::string FakeQuantizeAndTwoOutputBranchesWithConvolutionTransformation::getTestCaseName(
    const testing::TestParamInfo<FakeQuantizeAndTwoOutputBranchesWithConvolutionParams>& obj) {
    auto [netPrecision, inputShape, device, testValues] = obj.param;
    std::ostringstream result;
    result << netPrecision << "_" << inputShape << "_"
           << device << "_" << testValues.fqOnData << "_"
           << testValues.fqOnWeights1 << "_" << testValues.fqOnWeights2;
    return result.str();
}

void FakeQuantizeAndTwoOutputBranchesWithConvolutionTransformation::SetUp() {
    auto [netPrecision, inputShape, device, testValues] = this->GetParam();
    targetDevice = device;

    init_input_shapes(inputShape);

    function = ov::builder::subgraph::FakeQuantizeAndTwoOutputBranchesWithConvolutionFunction::getOriginal(
        netPrecision,
        inputShape,
        testValues.fqOnData,
        testValues.fqOnWeights1,
        testValues.fqOnWeights2);
}

TEST_P(FakeQuantizeAndTwoOutputBranchesWithConvolutionTransformation, CompareWithRefImpl) {
    run();
};

}  // namespace LayerTestsDefinitions
