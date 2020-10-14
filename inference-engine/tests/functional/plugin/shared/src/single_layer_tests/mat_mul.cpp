// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <functional>
#include <memory>
#include <string>
#include <tuple>
#include <vector>

#include "single_layer_tests/mat_mul.hpp"
#include "ngraph_functions/builders.hpp"

namespace LayerTestsDefinitions {

std::string MatMulTest::getTestCaseName(const testing::TestParamInfo<MatMulLayerTestParamsSet> &obj) {
    InferenceEngine::Precision netPrecision;
    InferenceEngine::Precision inPrc, outPrc;
    InferenceEngine::Layout inLayout;
    ShapeRelatedParams shapeRelatedParams;
    ngraph::helpers::InputLayerType secondaryInputType;
    std::string targetDevice;
    std::map<std::string, std::string> additionalConfig;
    std::tie(shapeRelatedParams, netPrecision, inPrc, outPrc, inLayout, secondaryInputType, targetDevice, additionalConfig) =
        obj.param;

    std::ostringstream result;
    result << "IS0=" << CommonTestUtils::vec2str(shapeRelatedParams.firstInputShape) << "_";
    result << "IS1=" << CommonTestUtils::vec2str(shapeRelatedParams.secondInputShape) << "_";
    result << "transpose_a=" << shapeRelatedParams.transposeA << "_";
    result << "transpose_b=" << shapeRelatedParams.transposeB << "_";
    result << "secondaryInputType=" << secondaryInputType << "_";
    result << "netPRC=" << netPrecision.name() << "_";
    result << "inPRC=" << inPrc.name() << "_";
    result << "outPRC=" << outPrc.name() << "_";
    result << "inL=" << inLayout << "_";
    result << "trgDev=" << targetDevice;
    return result.str();
}

void MatMulTest::SetUp() {
    ShapeRelatedParams shapeRelatedParams;
    ngraph::helpers::InputLayerType secondaryInputType;
    auto netPrecision = InferenceEngine::Precision::UNSPECIFIED;
    std::map<std::string, std::string> additionalConfig;
    std::tie(shapeRelatedParams, netPrecision, inPrc, outPrc, inLayout, secondaryInputType, targetDevice, additionalConfig) =
        this->GetParam();

    configuration.insert(additionalConfig.begin(), additionalConfig.end());

    auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);
    auto params = ngraph::builder::makeParams(ngPrc, {shapeRelatedParams.firstInputShape});

    auto secondaryInput = ngraph::builder::makeInputLayer(ngPrc, secondaryInputType, shapeRelatedParams.secondInputShape);
    if (secondaryInputType == ngraph::helpers::InputLayerType::PARAMETER) {
        params.push_back(std::dynamic_pointer_cast<ngraph::opset3::Parameter>(secondaryInput));
    }
    auto paramOuts = ngraph::helpers::convert2OutputVector(
            ngraph::helpers::castOps2Nodes<ngraph::op::Parameter>(params));
    auto MatMul = std::dynamic_pointer_cast<ngraph::opset3::MatMul>(
            ngraph::builder::makeMatMul(paramOuts[0], secondaryInput, shapeRelatedParams.transposeA, shapeRelatedParams.transposeB));
    ngraph::ResultVector results{std::make_shared<ngraph::opset1::Result>(MatMul)};
    function = std::make_shared<ngraph::Function>(results, params, "MatMul");
}

TEST_P(MatMulTest, CompareWithRefs) {
    Run();
};

}  // namespace LayerTestsDefinitions
