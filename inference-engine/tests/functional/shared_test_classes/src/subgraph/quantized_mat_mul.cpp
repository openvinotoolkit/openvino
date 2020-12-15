// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/subgraph/quantized_mat_mul.hpp"
#include "ngraph_functions/builders.hpp"

namespace SubgraphTestsDefinitions {

using ngraph::helpers::QuantizationGranularity;

std::string QuantMatMulTest::getTestCaseName(const testing::TestParamInfo<QuantMatMulLayerTestParamsSet> &obj) {
    QuantParams quantParams;
    InferenceEngine::Precision netPrecision;
    InferenceEngine::SizeVector inputShape0;
    InferenceEngine::SizeVector inputShape1;
    std::string targetDevice;
    std::tie(quantParams, netPrecision, inputShape0, inputShape1, targetDevice) = obj.param;

    size_t quantLevels;
    QuantizationGranularity quantGranularity;
    InferenceEngine::Precision fqPrec0;
    std::tie(quantLevels, quantGranularity, fqPrec0) = quantParams;

    std::ostringstream result;
    result << "IS0=" << CommonTestUtils::vec2str(inputShape0) << "_";
    result << "IS1=" << CommonTestUtils::vec2str(inputShape1) << "_";
    result << "Levels=" << quantLevels << "_";
    result << "QuantGranularity=" << quantGranularity << "_";
    result << "fq0PRC=" << fqPrec0.name() << "_";
    result << "netPRC=" << netPrecision.name() << "_";
    result << "targetDevice=" << targetDevice;
    return result.str();
}

void QuantMatMulTest::SetUp() {
    QuantParams quantParams;
    InferenceEngine::SizeVector inputShape0;
    InferenceEngine::SizeVector inputShape1;
    auto netPrecision = InferenceEngine::Precision::UNSPECIFIED;
    std::tie(quantParams, netPrecision, inputShape0, inputShape1, targetDevice) = this->GetParam();

    size_t quantLevels;
    QuantizationGranularity quantGranularity;
    InferenceEngine::Precision fqPrec0;
    std::tie(quantLevels, quantGranularity, fqPrec0) = quantParams;

    auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);
    auto params = ngraph::builder::makeParams(ngPrc, {inputShape0, inputShape1});
    auto paramOuts = ngraph::helpers::convert2OutputVector(
            ngraph::helpers::castOps2Nodes<ngraph::op::Parameter>(params));

    auto makeFakeQuantizeNode = [ngPrc, quantLevels, quantGranularity](const ngraph::Output<ngraph::Node> &in,
            std::vector<size_t> inputShape, InferenceEngine::Precision prec) -> std::shared_ptr<ngraph::Node> {
        std::vector<size_t> dataFqConstShapes(inputShape.size(), 1);
        if (quantGranularity == ngraph::helpers::Perchannel)
            dataFqConstShapes[1] = inputShape[1];
        size_t constDataSize = ngraph::shape_size(dataFqConstShapes);
        std::vector<float> inputLowData(constDataSize), inputHighData(constDataSize), outputLowData(constDataSize), outputHighData(constDataSize);
        for (int i = 0; i < constDataSize; i++) {
            inputLowData[i] = 0;
            inputHighData[i] = 255;
            outputLowData[i] = prec == InferenceEngine::Precision::I8 ? -128 : 0;
            outputHighData[i] = prec == InferenceEngine::Precision::I8 ? 127 : 255;
        }
        return ngraph::builder::makeFakeQuantize(in, ngPrc, quantLevels, dataFqConstShapes, inputLowData, inputHighData, outputLowData, outputHighData);
    };

    auto dataFq0 = makeFakeQuantizeNode(paramOuts[0], inputShape0, fqPrec0);
    auto dataFq1 = makeFakeQuantizeNode(paramOuts[1], inputShape1, InferenceEngine::Precision::I8);

    auto MatMul = std::dynamic_pointer_cast<ngraph::opset3::MatMul>(
            ngraph::builder::makeMatMul(dataFq0, dataFq1));
    ngraph::ResultVector results{std::make_shared<ngraph::opset1::Result>(MatMul)};
    function = std::make_shared<ngraph::Function>(results, params, "QuantMatMul");
}
}  // namespace SubgraphTestsDefinitions
