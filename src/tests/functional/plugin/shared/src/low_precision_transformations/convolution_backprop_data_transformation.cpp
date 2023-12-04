// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision_transformations/convolution_backprop_data_transformation.hpp"

#include <tuple>
#include <vector>
#include <string>

#include "ov_lpt_models/convolution_backprop_data.hpp"

namespace LayerTestsDefinitions {

std::string ConvolutionBackpropDataTransformation::getTestCaseName(const testing::TestParamInfo<ConvolutionBackpropDataTransformationParams>& obj) {
    ngraph::element::Type netPrecision;
    std::pair<ngraph::PartialShape, bool> inputShape;
    ngraph::Shape outputShape;
    std::string targetDevice;
    ov::pass::low_precision::LayerTransformation::Params params;
    ConvolutionBackpropDataTransformationParam param;
    std::tie(netPrecision, inputShape, outputShape, targetDevice, params, param) = obj.param;

    std::ostringstream result;
    result << getTestCaseNameByParams(netPrecision, inputShape.first, targetDevice, params) << "_" <<
        outputShape << "_" <<
        param.fakeQuantizeOnData << "_" <<
        param.fakeQuantizeOnWeights << "_" <<
        param.dequantizationOnWeights;
    return result.str();
}

void ConvolutionBackpropDataTransformation::SetUp() {
    threshold = 0.1f;

    ngraph::element::Type netPrecision;
    std::pair<ngraph::PartialShape, bool> inputShapeAndHandling;
    ngraph::Shape outputShape;
    ov::pass::low_precision::LayerTransformation::Params params;
    ConvolutionBackpropDataTransformationParam param;
    std::tie(netPrecision, inputShapeAndHandling, outputShape, targetDevice, params, param) = this->GetParam();

    std::shared_ptr<ngraph::Node> weights;

    const auto inputShape = inputShapeAndHandling.first;
    ngraph::Shape weightsShape(4, 1ul);
    weightsShape[0] = inputShape[1].get_length();
    weightsShape[1] = inputShape[1].get_length() / 2;

    if (!param.fakeQuantizeOnWeights.empty()) {
        weights = ngraph::builder::subgraph::ConvolutionBackpropDataFunction::getWeights(
            weightsShape,
            netPrecision,
            param.fakeQuantizeOnWeights);
    } else {
        weights = ngraph::builder::subgraph::ConvolutionBackpropDataFunction::getWeights(
            weightsShape,
            netPrecision,
            param.dequantizationOnWeights);
    }

    function = ngraph::builder::subgraph::ConvolutionBackpropDataFunction::get(
        netPrecision,
        inputShape,
        outputShape,
        param.fakeQuantizeOnData,
        weights);
}

void ConvolutionBackpropDataTransformation::Run() {
    LayerTestsCommon::Run();

    const auto inputShape = std::get<1>(GetParam());
    if (inputShape.second) {
        const auto params = std::get<5>(GetParam());
        const auto actualType = getRuntimePrecision(params.layerName);
        EXPECT_EQ(actualType, params.expectedKernelType);
    }
}

TEST_P(ConvolutionBackpropDataTransformation, CompareWithRefImpl) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();
    Run();
};

}  // namespace LayerTestsDefinitions
