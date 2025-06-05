// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision_transformations/convolution_transformation.hpp"

#include <memory>
#include <tuple>
#include <vector>
#include <string>


#include "common_test_utils/common_utils.hpp"
#include "ov_lpt_models/fake_quantize_and_convolution.hpp"

namespace LayerTestsDefinitions {

std::string ConvolutionTransformation::getTestCaseName(const testing::TestParamInfo<ConvolutionTransformationParams>& obj) {
    auto [netPrecision, inputShape, device, param] = obj.param;
    ov::pass::low_precision::LayerTransformation::Params params;
    std::ostringstream result;
    result << get_test_case_name_by_params(netPrecision, inputShape, device, params) <<
           "_rank=" << inputShape.size() <<
        "D_fq_on_data={" << param.fakeQuantizeOnData <<
        "}_fq_on_weights={" << param.fakeQuantizeOnWeights << "}";
    return result.str();
}

void ConvolutionTransformation::SetUp() {
    auto [netPrecision, inputShape, device, param] = this->GetParam();
    targetDevice = device;

    init_input_shapes(inputShape);

    function = ov::builder::subgraph::FakeQuantizeAndConvolutionFunction::get(
        netPrecision,
        inputShape,
        param.fakeQuantizeOnData,
        param.fakeQuantizeOnWeights);
}

void ConvolutionTransformation::run() {
    LayerTransformation::run();

    const auto params = std::get<3>(GetParam());
    const auto actualPrecision = get_runtime_precision_by_type(params.layerName);
    auto expectedPrecision = params.expectedKernelType;
    if (expectedPrecision == "f32" && std::get<0>(GetParam()) == ov::element::f16) {
        expectedPrecision = "f16";
    }
    EXPECT_EQ(actualPrecision, expectedPrecision);
}

TEST_P(ConvolutionTransformation, CompareWithRefImpl) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();
    run();
};

}  // namespace LayerTestsDefinitions
