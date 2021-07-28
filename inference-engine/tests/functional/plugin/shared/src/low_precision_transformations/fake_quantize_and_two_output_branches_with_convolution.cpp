// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision_transformations/fake_quantize_and_two_output_branches_with_convolution.hpp"

#include <memory>
#include <tuple>
#include <vector>
#include <string>

#include <ie_core.hpp>

#include "common_test_utils/common_utils.hpp"
#include "functional_test_utils/plugin_cache.hpp"
#include "shared_test_classes/base/layer_test_utils.hpp"
#include "functional_test_utils/blob_utils.hpp"
#include "ngraph_functions/pass/convert_prc.hpp"

namespace LayerTestsDefinitions {

std::string FakeQuantizeAndTwoOutputBranchesWithConvolutionTransformation::getTestCaseName(
    testing::TestParamInfo<FakeQuantizeAndTwoOutputBranchesWithConvolutionParams> obj) {
    ngraph::element::Type netPrecision;
    ngraph::PartialShape inputShape;
    std::string targetDevice;
    ngraph::pass::low_precision::LayerTransformation::Params params;
    FakeQuantizeAndTwoOutputBranchesWithConvolution testValues;
    std::tie(netPrecision, inputShape, targetDevice, params, testValues) = obj.param;

    std::ostringstream result;
    result << netPrecision << "_" << inputShape << "_"
           << targetDevice << "_" << testValues.fqOnData << "_"
           << testValues.fqOnWeights1 << "_" << testValues.fqOnWeights2;
    return result.str();
}

void FakeQuantizeAndTwoOutputBranchesWithConvolutionTransformation::SetUp() {
    threshold = 0.1f;
    ngraph::element::Type netPrecision;
    ngraph::PartialShape inputShape;
    ngraph::pass::low_precision::LayerTransformation::Params params;
    FakeQuantizeAndTwoOutputBranchesWithConvolution testValues;
    std::tie(netPrecision, inputShape, targetDevice, params, testValues) = this->GetParam();

    function = ngraph::builder::subgraph::FakeQuantizeAndTwoOutputBranchesWithConvolutionFunction::getOriginal(
        netPrecision,
        inputShape,
        testValues.fqOnData,
        testValues.fqOnWeights1,
        testValues.fqOnWeights2);
}

TEST_P(FakeQuantizeAndTwoOutputBranchesWithConvolutionTransformation, CompareWithRefImpl) {
    Run();
};

}  // namespace LayerTestsDefinitions
