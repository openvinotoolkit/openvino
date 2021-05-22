// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision_transformations/fake_quantize_transformation.hpp"

#include <memory>
#include <tuple>
#include <vector>
#include <string>
#include <ie_core.hpp>

#include <transformations/init_node_info.hpp>

#include "low_precision/fuse_subtract_to_fake_quantize.hpp"
#include "low_precision/fuse_multiply_to_fake_quantize.hpp"

namespace LayerTestsDefinitions {

std::string FakeQuantizeTransformation::getTestCaseName(testing::TestParamInfo<FakeQuantizeTransformationParams> obj) {
    ngraph::element::Type netPrecision;
    ngraph::Shape inputShape;
    std::string targetDevice;
    ngraph::pass::low_precision::LayerTransformation::Params params;
    FakeQuantizeTransformationParam testParams;
    std::tie(netPrecision, inputShape, targetDevice, params, testParams) = obj.param;

    std::ostringstream result;
    result << getTestCaseNameByParams(netPrecision, inputShape, targetDevice, params) << "_" << testParams.fakequantize;
    return result.str();
}

void FakeQuantizeTransformation::SetUp() {
    ngraph::element::Type netPrecision;
    ngraph::Shape inputShape;
    ngraph::pass::low_precision::LayerTransformation::Params params;
    FakeQuantizeTransformationParam testParams;
    std::tie(netPrecision, inputShape, targetDevice, params, testParams) = this->GetParam();

    function = ngraph::builder::subgraph::FakeQuantizeFunction::getOriginalWithMaxPool(
        netPrecision,
        inputShape,
        testParams.fakequantize);
}

void FakeQuantizeTransformation::Run() {
    LayerTestsCommon::Run();

    const auto params = std::get<4>(GetParam());
    const auto actualPrecision = getRuntimePrecisionByType(params.layerName);
    auto expectedPrecision = params.expectedKernelType;
    if (expectedPrecision == "FP32" && std::get<0>(GetParam()) == ngraph::element::f16) {
        expectedPrecision = "FP16";
    }
    EXPECT_EQ(actualPrecision, expectedPrecision);
}

TEST_P(FakeQuantizeTransformation, CompareWithRefImpl) {
    Run();
};

}  // namespace LayerTestsDefinitions
