// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision_transformations/multiply_to_group_convolution_transformation.hpp"

#include <memory>
#include <tuple>
#include <vector>
#include <string>


#include "common_test_utils/common_utils.hpp"

#include "ov_lpt_models/multiply_to_group_convolution.hpp"

namespace LayerTestsDefinitions {

std::string MultiplyToGroupConvolutionTransformation::getTestCaseName(const testing::TestParamInfo<MultiplyToGroupConvolutionTransformationParams>& obj) {
    std::string targetDevice;
    ov::element::Type precision;
    ov::PartialShape shape;
    auto params = LayerTestsUtils::LayerTransformationParamsNGraphFactory::createParamsU8I8();
    MultiplyToGroupConvolutionTransformationParam param;
    std::tie(precision, shape, targetDevice, param) = obj.param;

    std::ostringstream result;
    result << get_test_case_name_by_params(precision, shape, targetDevice, params) << "_" <<
           param.fqOnData << "_" <<
        param.constant << "_" <<
        param.layerName << "_" <<
        param.expectedKernelType << "_" <<
        param.parentHasOneConsumer;
    return result.str();
}

void MultiplyToGroupConvolutionTransformation::SetUp() {
    ov::PartialShape shape;
    ov::element::Type precision;
    MultiplyToGroupConvolutionTransformationParam param;
    std::tie(precision, shape, targetDevice, param) = this->GetParam();

    init_input_shapes(shape);

    function = ov::builder::subgraph::MultiplyToGroupConvolutionFunction::getOriginal(
        precision,
        shape,
        param.fqOnData,
        param.constant,
        param.parentHasOneConsumer);
}

void MultiplyToGroupConvolutionTransformation::run() {
    LayerTransformation::run();

    const auto param = std::get<3>(GetParam());
    const auto actualPrecision = get_runtime_precision(param.layerName);
    auto expectedPrecision = param.expectedKernelType;
    if (expectedPrecision == "FP32" && std::get<0>(GetParam()) == ov::element::f16) {
        expectedPrecision = "FP16";
    }
    EXPECT_EQ(actualPrecision, expectedPrecision);
}

TEST_P(MultiplyToGroupConvolutionTransformation, CompareWithRefImpl) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();
    run();
};

}  // namespace LayerTestsDefinitions
