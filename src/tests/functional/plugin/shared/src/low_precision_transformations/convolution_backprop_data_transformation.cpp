// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision_transformations/convolution_backprop_data_transformation.hpp"

#include <tuple>
#include <vector>
#include <string>

#include "ov_lpt_models/convolution_backprop_data.hpp"

namespace LayerTestsDefinitions {

std::string ConvolutionBackpropDataTransformation::getTestCaseName(const testing::TestParamInfo<ConvolutionBackpropDataTransformationParams>& obj) {
    auto [netPrecision, inputShape, outputShape, device, param] = obj.param;

    std::ostringstream result;
    result << get_test_case_name_by_params(netPrecision, inputShape.first, device) << "_" <<
           outputShape << "_" <<
        param.fakeQuantizeOnData << "_" <<
        param.fakeQuantizeOnWeights << "_" <<
        param.dequantizationOnWeights;
    return result.str();
}

void ConvolutionBackpropDataTransformation::SetUp() {
    auto [netPrecision, inputShapeAndHandling, outputShape, device, param] = this->GetParam();
    targetDevice = device;

    std::shared_ptr<ov::Node> weights;

    const auto& inputShape = inputShapeAndHandling.first;
    const auto rank = inputShape.rank();
    init_input_shapes(inputShape);

    ov::Shape weightsShape(rank.get_length(), 1ul);
    weightsShape[0] = inputShape[1].get_length();
    weightsShape[1] = inputShape[1].get_length() / 2;

    if (!param.fakeQuantizeOnWeights.empty()) {
        weights = ov::builder::subgraph::ConvolutionBackpropDataFunction::getWeights(
            weightsShape,
            netPrecision,
            param.fakeQuantizeOnWeights);
    } else {
        weights = ov::builder::subgraph::ConvolutionBackpropDataFunction::getWeights(
            weightsShape,
            netPrecision,
            param.dequantizationOnWeights);
    }

    function = ov::builder::subgraph::ConvolutionBackpropDataFunction::get(
        netPrecision,
        inputShape,
        outputShape,
        param.fakeQuantizeOnData,
        weights);
}

void ConvolutionBackpropDataTransformation::run() {
    LayerTransformation::run();

    const auto inputShape = std::get<1>(GetParam());
    if (inputShape.second) {
        const auto params = std::get<4>(GetParam());
        const auto actualType = get_runtime_precision(params.layerName);
        EXPECT_EQ(actualType, params.expectedKernelType);
    }
}

TEST_P(ConvolutionBackpropDataTransformation, CompareWithRefImpl) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();
    run();
};

}  // namespace LayerTestsDefinitions
