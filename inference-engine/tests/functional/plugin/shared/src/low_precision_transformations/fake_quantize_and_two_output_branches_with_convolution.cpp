// Copyright (C) 2020 Intel Corporation
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
    ngraph::Shape inputShape;
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
    ngraph::Shape inputShape;
    ngraph::pass::low_precision::LayerTransformation::Params params;
    FakeQuantizeAndTwoOutputBranchesWithConvolution testValues;
    std::tie(netPrecision, inputShape, targetDevice, params, testValues) = this->GetParam();

    function = ngraph::builder::subgraph::FakeQuantizeAndTwoOutputBranchesWithConvolutionFunction::getOriginal(
        netPrecision,
        inputShape,
        testValues.fqOnData,
        testValues.fqOnWeights1,
        testValues.fqOnWeights2);

    validate();
}

void FakeQuantizeAndTwoOutputBranchesWithConvolutionTransformation::validate() {
    ngraph::element::Type precision;
    ngraph::Shape inputShapes;
    std::string targetDevice;
    ngraph::pass::low_precision::LayerTransformation::Params params;
    FakeQuantizeAndTwoOutputBranchesWithConvolution testValues;
    std::tie(precision, inputShapes, targetDevice, params, testValues) = this->GetParam();

    const auto transformed = transformNGraph(params, getLowPrecisionTransformationsNGraph(params));
    EXPECT_EQ(1ul, transformed->get_output_size());

    const auto output = transformed->get_output_op(0);
    const auto concat = output->get_input_node_shared_ptr(0);

    const std::string typeName = concat->get_type_name();
    ASSERT_EQ("Concat", typeName);

    EXPECT_EQ(2ul, concat->get_input_size());
    for (size_t i = 0; i < 2; ++i) {
        const auto scaleShift = concat->get_input_node_shared_ptr(i);
        const std::string scaleShiftName = scaleShift->get_type_name();
        ASSERT_EQ("ScaleShiftIE", scaleShiftName);
    }
}

TEST_P(FakeQuantizeAndTwoOutputBranchesWithConvolutionTransformation, CompareWithRefImpl) {
    Run();
};

}  // namespace LayerTestsDefinitions
