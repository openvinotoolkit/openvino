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
std::ostream& operator<<(std::ostream & os, MatMulParams::InputLayerType type) {
    switch (type) {
        case MatMulParams::InputLayerType::CONSTANT:
            os << "CONSTANT";
            break;
        case MatMulParams::InputLayerType::PARAMETER:
            os << "PARAMETER";
            break;
        default:
            THROW_IE_EXCEPTION << "NOT_SUPPORTED_INPUT_LAYER_TYPE";
    }
    return os;
}

std::string MatMulTest::getTestCaseName(const testing::TestParamInfo<MatMulLayerTestParamsSet> &obj) {
    InferenceEngine::Precision netPrecision;
    InferenceEngine::SizeVector inputShape0;
    InferenceEngine::SizeVector inputShape1;
    MatMulParams::InputLayerType secondaryInputType;
    std::string targetDevice;
    std::tie(netPrecision, inputShape0, inputShape1, secondaryInputType, targetDevice) = obj.param;

    std::ostringstream result;
    result << "IS0=" << CommonTestUtils::vec2str(inputShape0) << "_";
    result << "IS1=" << CommonTestUtils::vec2str(inputShape1) << "_";
    result << "secondaryInputType=" << secondaryInputType << "_";
    result << "netPRC=" << netPrecision.name() << "_";
    result << "targetDevice=" << targetDevice;
    return result.str();
}

void MatMulTest::SetUp() {
    InferenceEngine::SizeVector inputShape0;
    InferenceEngine::SizeVector inputShape1;
    MatMulParams::InputLayerType secondaryInputType;
    auto netPrecision = InferenceEngine::Precision::UNSPECIFIED;
    std::tie(netPrecision, inputShape0, inputShape1, secondaryInputType, targetDevice) = this->GetParam();
    auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);
    auto params = ngraph::builder::makeParams(ngPrc, {inputShape0});

    std::shared_ptr<ngraph::Node> secondaryInput;
    switch (secondaryInputType) {
        case MatMulParams::InputLayerType::CONSTANT: {
            std::vector<float> data;
            data.resize(ngraph::shape_size(inputShape1));
            CommonTestUtils::fill_data_sine(data.data(), data.size(), 0, 10, 1);
            secondaryInput = ngraph::builder::makeConstant(ngPrc, inputShape1, data);
            break;
        }
        case MatMulParams::InputLayerType::PARAMETER:
            params.push_back(ngraph::builder::makeParams(ngPrc, {inputShape1})[0]);
            secondaryInput = params[1];
            break;
        default:
            FAIL() << "Unsupported secondaryInputType";
    }
    auto paramOuts = ngraph::helpers::convert2OutputVector(
            ngraph::helpers::castOps2Nodes<ngraph::op::Parameter>(params));
    auto MatMul = std::dynamic_pointer_cast<ngraph::opset3::MatMul>(
            ngraph::builder::makeMatMul(paramOuts[0], secondaryInput));
    ngraph::ResultVector results{std::make_shared<ngraph::opset1::Result>(MatMul)};
    function = std::make_shared<ngraph::Function>(results, params, "MatMul");
}

TEST_P(MatMulTest, CompareWithRefs) {
    Run();
};

}  // namespace LayerTestsDefinitions
