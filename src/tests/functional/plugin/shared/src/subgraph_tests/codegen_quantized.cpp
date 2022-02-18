
// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <memory>
#include <tuple>
#include <vector>
#include <string>

#include <ie_core.hpp>
#include "subgraph_tests/codegen_quantized.hpp"
#include "ngraph_ops/type_relaxed.hpp"

namespace LayerTestsDefinitions {

std::string CodegenQuantized::getTestCaseName(testing::TestParamInfo<LayerTestsDefinitions::inputParams> obj) {
    InferenceEngine::Precision netPrecision;
    InferenceEngine::SizeVector inputShapes, newInputShapes;
    ov::element::Type inputPrecision;
    ov::element::Type outputPrecision;
    std::string targetDevice;
    std::tie(netPrecision, inputShapes, inputPrecision, outputPrecision, targetDevice) = obj.param;

    std::ostringstream result;
    result << "IS=" << CommonTestUtils::vec2str(inputShapes) << "_";
    result << "netPRC=" << netPrecision.name() << "_";
    result << "D=" << targetDevice << "_";
    result << "IN=" << inputPrecision << "_";
    result << "OUT=" << outputPrecision;
    return result.str();
}

void CodegenQuantized::GenerateInputs() {
    inputs.clear();
    const auto& inputsInfo = executableNetwork.GetInputsInfo();
    const auto& functionParams = function->get_parameters();
    for (int i = 0; i < functionParams.size(); ++i) {
        const auto& param = functionParams[i];
        const bool is_signed = param->output(0).get_element_type().is_signed();

        const auto infoIt = inputsInfo.find(param->get_friendly_name());
        GTEST_ASSERT_NE(infoIt, inputsInfo.cend());
        InferenceEngine::InputInfo::CPtr info = infoIt->second;
        InferenceEngine::Blob::Ptr blob = GenerateInput(*info);
        float* rawBlobDataPtr = blob->buffer().as<float*>();
        const size_t blobSize = blob->size();
        float value = 0.f;
        for (size_t i = 0; i < blobSize; i++) {
            rawBlobDataPtr[i] = value;
            value += value + 1.f/256.f;
            if (value > 1.f) {
                value = 0.f;
            }
        }
        inputs.push_back(blob);
    }
}

void CodegenQuantized::SetUp() {
    std::vector<size_t> inputShape0;
    InferenceEngine::Precision netPrecision;
    ov::element::Type inputPrecision;
    ov::element::Type outputPrecision;
    std::tie(netPrecision, inputShape0, inputPrecision, outputPrecision, targetDevice) = this->GetParam();

    const auto input = std::make_shared<ngraph::opset1::Parameter>(inputPrecision, ngraph::Shape{inputShape0});
    const auto min = std::make_shared<ngraph::opset1::Constant>(inputPrecision, ov::Shape{}, std::vector<float>{0.f});
    const auto max = std::make_shared<ngraph::opset1::Constant>(inputPrecision, ov::Shape{}, std::vector<float>{1.f});
    const auto fakeQuantize = std::make_shared<ngraph::opset1::FakeQuantize>(input, min, max, min, max, 256ul);
    const auto slope = std::make_shared<ngraph::opset1::Constant>(ov::element::f32, ov::Shape{}, std::vector<float>{0.f});
    const auto prelu = std::make_shared<ngraph::opset1::PRelu>(fakeQuantize, slope);
    const auto result = std::make_shared<ngraph::opset1::Result>(prelu);
    function = std::make_shared<ngraph::Function>(ngraph::ResultVector{result}, ngraph::ParameterVector{input}, "CodegenQuantized");
}

TEST_P(CodegenQuantized, CompareWithRefImpl) {
    Run();
};

}  // namespace LayerTestsDefinitions
