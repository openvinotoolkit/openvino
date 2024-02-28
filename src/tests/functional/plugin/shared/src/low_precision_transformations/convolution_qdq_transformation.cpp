// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision_transformations/convolution_qdq_transformation.hpp"

#include <memory>
#include <tuple>
#include <vector>
#include <string>


#include "common_test_utils/common_utils.hpp"
#include "ov_lpt_models/fake_quantize_and_convolution.hpp"

namespace LayerTestsDefinitions {

std::string ConvolutionQDqTransformation::getTestCaseName(const testing::TestParamInfo<ConvolutionQDqTransformationParams>& obj) {
    ov::element::Type netPrecision;
    ov::PartialShape inputShape;
    std::string targetDevice;
    ov::pass::low_precision::LayerTransformation::Params params;
    ConvolutionQDqTransformationParam param;
    std::tie(netPrecision, inputShape, targetDevice, params, param) = obj.param;

    std::ostringstream result;
    result << get_test_case_name_by_params(netPrecision, inputShape, targetDevice, params) << param;
    return result.str();
}

void ConvolutionQDqTransformation::SetUp() {
    ov::element::Type netPrecision;
    ov::PartialShape inputShape;
    ov::pass::low_precision::LayerTransformation::Params params;
    ConvolutionQDqTransformationParam param;
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
        {});

    this->configuration[ov::hint::inference_precision.name()] = "f32";
}

void ConvolutionQDqTransformation::run() {
    LayerTransformation::run();

    const auto params = std::get<4>(GetParam());
    const auto actualType = get_runtime_precision_by_type(params.layerName);
    EXPECT_EQ(actualType, params.expectedKernelType);
}

TEST_P(ConvolutionQDqTransformation, CompareWithRefImpl) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();
    run();
};

}  // namespace LayerTestsDefinitions
