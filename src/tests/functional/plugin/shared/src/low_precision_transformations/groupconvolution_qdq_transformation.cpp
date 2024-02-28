// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision_transformations/groupconvolution_qdq_transformation.hpp"

#include <memory>
#include <tuple>
#include <vector>
#include <string>


#include "common_test_utils/common_utils.hpp"
#include "ov_lpt_models/fake_quantize_and_convolution.hpp"

namespace LayerTestsDefinitions {

std::string GroupConvolutionQDqTransformation::getTestCaseName(const testing::TestParamInfo<GroupConvolutionQDqTransformationParams>& obj) {
    ov::element::Type netPrecision;
    ov::PartialShape inputShape;
    std::string targetDevice;
    ov::pass::low_precision::LayerTransformation::Params params;
    GroupConvolutionQDqTransformationParam param;
    std::tie(netPrecision, inputShape, targetDevice, params, param) = obj.param;

    std::ostringstream result;
    result << get_test_case_name_by_params(netPrecision, inputShape, targetDevice, params) << param;
    return result.str();
}

void GroupConvolutionQDqTransformation::SetUp() {
    ov::element::Type netPrecision;
    ov::PartialShape inputShape;
    ov::pass::low_precision::LayerTransformation::Params params;
    GroupConvolutionQDqTransformationParam param;
    std::tie(netPrecision, inputShape, targetDevice, params, param) = this->GetParam();

    init_input_shapes(inputShape);

    function = ov::builder::subgraph::FakeQuantizeAndConvolutionFunction::get(
        netPrecision,
        inputShape,
        param.fakeQuantizeOnData,
        param.convertOnData,
        param.dequantizationOnData,
        param.constantOnWeights,
        param.fakeQuantizeOnWeights,
        param.convertOnWeights,
        param.dequantizationOnWeights,
        {}, {}, {}, param.reshape, {}, "GroupConvolution", param.multiplyAfter);
}

void GroupConvolutionQDqTransformation::run() {
    LayerTransformation::run();

    const auto params = std::get<4>(GetParam());
    const auto actualType = get_runtime_precision(params.layerName);
    EXPECT_EQ(actualType, params.expectedKernelType);
}

TEST_P(GroupConvolutionQDqTransformation, CompareWithRefImpl) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();
    run();
};

}  // namespace LayerTestsDefinitions
