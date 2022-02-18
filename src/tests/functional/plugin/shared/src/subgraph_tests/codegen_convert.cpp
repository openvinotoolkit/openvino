
// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <memory>
#include <tuple>
#include <vector>
#include <string>

#include <ie_core.hpp>
#include "subgraph_tests/codegen_convert.hpp"
#include "ngraph_ops/type_relaxed.hpp"

namespace LayerTestsDefinitions {

std::string CodegenConvert::getTestCaseName(testing::TestParamInfo<LayerTestsDefinitions::inputParams> obj) {
    ov::element::Type netPrecision;
    InferenceEngine::SizeVector inputShapes, newInputShapes;
    std::pair<ov::element::Type, ov::element::Type> precisions;
    std::string targetDevice;
    std::tie(netPrecision, inputShapes, precisions, targetDevice) = obj.param;

    std::ostringstream result;
    result << "IS=" << CommonTestUtils::vec2str(inputShapes) << "_";
    result << "netPRC=" << netPrecision << "_";
    result << "D=" << targetDevice << "_";
    result << "IN=" << precisions.first << "_";
    result << "OUT=" << precisions.second;
    return result.str();
}

void CodegenConvert::GenerateInputs() {
    std::vector<size_t> inputShape0;
    ov::element::Type netPrecision;
    std::pair<ov::element::Type, ov::element::Type> convertPrecisions;
    std::tie(netPrecision, inputShape0, convertPrecisions, targetDevice) = this->GetParam();

    inputs.clear();
    const auto& inputsInfo = executableNetwork.GetInputsInfo();
    const auto& functionParams = function->get_parameters();
    for (int i = 0; i < functionParams.size(); ++i) {
        const auto& param = functionParams[i];

        const auto type = param->output(0).get_element_type();
        const auto infoIt = inputsInfo.find(param->get_friendly_name());
        GTEST_ASSERT_NE(infoIt, inputsInfo.cend());
        InferenceEngine::InputInfo::CPtr info = infoIt->second;
        InferenceEngine::Blob::Ptr blob = GenerateInput(*info);
        char* rawBlobDataPtr = blob->buffer().as<char*>();
        const size_t blobSize = blob->size();
        int value = 0;
        for (size_t i = 0; i < blobSize; i++, value++) {
            if (value > 255) {
                value = 0;
            }

            switch (convertPrecisions.first) {
                case ov::element::i8: {
                    rawBlobDataPtr[i] = value - 128;
                    break;
                }
                case ov::element::u8: {
                    rawBlobDataPtr[i] = value;
                    break;
                }
                case ov::element::f32: {
                    auto raw = (float*)rawBlobDataPtr;
                    raw[i] = static_cast<float>(i);
                    break;
                }
            }
        }
        inputs.push_back(blob);
    }
}

void CodegenConvert::SetUp() {
    std::vector<size_t> inputShape0;
    ov::element::Type netPrecision;
    std::pair<ov::element::Type, ov::element::Type> convertPrecisions;
    std::tie(netPrecision, inputShape0, convertPrecisions, targetDevice) = this->GetParam();

    const auto input = std::make_shared<ngraph::opset1::Parameter>(convertPrecisions.first, ngraph::Shape{inputShape0});
    input->set_friendly_name("input");

    const auto convert = std::make_shared<ngraph::opset1::Convert>(input, convertPrecisions.second);
    convert->set_friendly_name("convert");

    const auto original_subtract = std::make_shared<ngraph::opset1::Subtract>(
        convert,
        std::make_shared<ngraph::opset1::Constant>(convertPrecisions.second, ov::Shape{}, std::vector<float>{3.f}));

    const auto subtract = convertPrecisions.second == netPrecision ?
        // net precision => net precision
        original_subtract :
        // low precision (u8/i8[/u16/i16/u32/i32...- not supported by CPU]) => net precision
        std::make_shared<ngraph::op::TypeRelaxed<ngraph::opset1::Subtract>>(*original_subtract, ov::element::f32);
    subtract->set_friendly_name("subtract");

    const auto multiply = std::make_shared<ngraph::opset1::Multiply>(
        subtract,
        std::make_shared<ngraph::opset1::Constant>(ov::element::f32, ov::Shape{}, std::vector<float>{2.f}));
    multiply->set_friendly_name("multiply");

    const auto result = std::make_shared<ngraph::opset1::Result>(multiply);
    result->set_friendly_name("result");

    function = std::make_shared<ngraph::Function>(
        ngraph::ResultVector{result},
        ngraph::ParameterVector{input},
        "CodegenConvert");
}

void CodegenConvert::Run() {
    LayerTestsCommon::Run();

    auto execGraph = getExecGraphInfoAsMap();
    EXPECT_EQ(3, execGraph.size());
    EXPECT_EQ("Input", execGraph.find("input")->second);
    EXPECT_EQ("Subgraph", execGraph.find("multiply")->second);
    EXPECT_EQ("Output", execGraph.find("result")->second);
}

TEST_P(CodegenConvert, CompareWithRefImpl) {
    Run();
};

}  // namespace LayerTestsDefinitions
