// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision_transformations/groupconvolution_qdq_transformation.hpp"

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

std::string GroupConvolutionQDqTransformation::getTestCaseName(const testing::TestParamInfo<GroupConvolutionQDqTransformationParams>& obj) {
    ngraph::element::Type netPrecision;
    ngraph::PartialShape inputShape;
    std::string targetDevice;
    ov::pass::low_precision::LayerTransformation::Params params;
    GroupConvolutionQDqTransformationParam param;
    std::tie(netPrecision, inputShape, targetDevice, params, param) = obj.param;

    std::ostringstream result;
    result << getTestCaseNameByParams(netPrecision, inputShape, targetDevice, params) << param;
    return result.str();
}

void GroupConvolutionQDqTransformation::SetUp() {
    // threshold = 0.1f;

    ngraph::element::Type netPrecision;
    ngraph::PartialShape inputShape;
    ov::pass::low_precision::LayerTransformation::Params params;
    GroupConvolutionQDqTransformationParam param;
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
        {}, {}, {}, param.reshape, {}, "GroupConvolution", param.multiplyAfter);
}

void GroupConvolutionQDqTransformation::Run() {
    LayerTestsCommon::Run();

    const auto params = std::get<4>(GetParam());
    const auto actualType = getRuntimePrecision(params.layerName);
    EXPECT_EQ(actualType, params.expectedKernelType);
}

TEST_P(GroupConvolutionQDqTransformation, CompareWithRefImpl) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();
    Run();
};

}  // namespace LayerTestsDefinitions
