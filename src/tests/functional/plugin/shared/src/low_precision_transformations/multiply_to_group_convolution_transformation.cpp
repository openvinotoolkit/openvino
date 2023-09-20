// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision_transformations/multiply_to_group_convolution_transformation.hpp"

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
#include "ov_lpt_models/multiply_to_group_convolution.hpp"

namespace LayerTestsDefinitions {

std::string MultiplyToGroupConvolutionTransformation::getTestCaseName(const testing::TestParamInfo<MultiplyToGroupConvolutionTransformationParams>& obj) {
    std::string targetDevice;
    ngraph::element::Type precision;
    ngraph::PartialShape shape;
    auto params = LayerTestsUtils::LayerTransformationParamsNGraphFactory::createParamsU8I8();
    MultiplyToGroupConvolutionTransformationParam param;
    std::tie(precision, shape, targetDevice, param) = obj.param;

    std::ostringstream result;
    result << getTestCaseNameByParams(precision, shape, targetDevice, params) << "_" <<
        param.fqOnData << "_" <<
        param.constant << "_" <<
        param.layerName << "_" <<
        param.expectedKernelType << "_" <<
        param.parentHasOneConsumer;
    return result.str();
}

void MultiplyToGroupConvolutionTransformation::SetUp() {
    ngraph::PartialShape shape;
    ngraph::element::Type precision;
    MultiplyToGroupConvolutionTransformationParam param;
    std::tie(precision, shape, targetDevice, param) = this->GetParam();

    function = ngraph::builder::subgraph::MultiplyToGroupConvolutionFunction::getOriginal(
        precision,
        shape,
        param.fqOnData,
        param.constant,
        param.parentHasOneConsumer);
}

void MultiplyToGroupConvolutionTransformation::Run() {
    LayerTestsCommon::Run();

    const auto param = std::get<3>(GetParam());
    const auto actualPrecision = getRuntimePrecision(param.layerName);
    auto expectedPrecision = param.expectedKernelType;
    if (expectedPrecision == "FP32" && std::get<0>(GetParam()) == ngraph::element::f16) {
        expectedPrecision = "FP16";
    }
    EXPECT_EQ(actualPrecision, expectedPrecision);
}

TEST_P(MultiplyToGroupConvolutionTransformation, CompareWithRefImpl) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();
    Run();
};

}  // namespace LayerTestsDefinitions
