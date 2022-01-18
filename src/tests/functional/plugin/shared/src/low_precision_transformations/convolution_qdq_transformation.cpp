// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision_transformations/convolution_qdq_transformation.hpp"

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
#include "lpt_ngraph_functions/fake_quantize_and_convolution_function.hpp"

namespace LayerTestsDefinitions {

std::string ConvolutionQDqTransformation::getTestCaseName(const testing::TestParamInfo<ConvolutionQDqTransformationParams>& obj) {
    ngraph::element::Type netPrecision;
    ngraph::PartialShape inputShape;
    std::string targetDevice;
    ngraph::pass::low_precision::LayerTransformation::Params params;
    ConvolutionQDqTransformationParam param;
    std::tie(netPrecision, inputShape, targetDevice, params, param) = obj.param;

    std::ostringstream result;
    result << getTestCaseNameByParams(netPrecision, inputShape, targetDevice, params) << param;
    return result.str();
}

void ConvolutionQDqTransformation::SetUp() {
    // threshold = 0.1f;

    ngraph::element::Type netPrecision;
    ngraph::PartialShape inputShape;
    ngraph::pass::low_precision::LayerTransformation::Params params;
    ConvolutionQDqTransformationParam param;
    std::tie(netPrecision, inputShape, targetDevice, params, param) = this->GetParam();

    function = ngraph::builder::subgraph::FakeQuantizeAndConvolutionFunction::get(
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
}

void ConvolutionQDqTransformation::Run() {
    LayerTestsCommon::Run();

    const auto params = std::get<4>(GetParam());
    const auto actualType = getRuntimePrecisionByType(params.layerName);
    EXPECT_EQ(actualType, params.expectedKernelType);
}

TEST_P(ConvolutionQDqTransformation, CompareWithRefImpl) {
    Run();
};

}  // namespace LayerTestsDefinitions
