
// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "subgraph_tests/codegen_quantized.hpp"

#include <memory>
#include <tuple>
#include <vector>
#include <string>

#include <ie_core.hpp>
#include "ngraph_ops/type_relaxed.hpp"

namespace LayerTestsDefinitions {

std::string CodegenQuantized::getTestCaseName(testing::TestParamInfo<LayerTestsDefinitions::inputParams> obj) {
    InferenceEngine::Precision netPrecision;
    InferenceEngine::SizeVector inputShapes, newInputShapes;
    ov::element::Type inputPrecision;
    bool modelInLowPrecision;
    std::string targetDevice;
    std::tie(netPrecision, inputShapes, inputPrecision, modelInLowPrecision, targetDevice) = obj.param;

    std::ostringstream result;
    result << "IS=" << CommonTestUtils::vec2str(inputShapes) << "_";
    result << "netPRC=" << netPrecision.name() << "_";
    result << "D=" << targetDevice << "_";
    result << "IN=" << inputPrecision << "_";
    result << "LP=" << modelInLowPrecision;
    return result.str();
}

void CodegenQuantized::GenerateInputs() {
    inputs.clear();
    const auto& inputsInfo = executableNetwork.GetInputsInfo();
    const auto& functionParams = function->get_parameters();
    for (int i = 0; i < functionParams.size(); ++i) {
        const auto& param = functionParams[i];
        const auto infoIt = inputsInfo.find(param->get_friendly_name());
        GTEST_ASSERT_NE(infoIt, inputsInfo.cend());
        InferenceEngine::InputInfo::CPtr info = infoIt->second;
        InferenceEngine::Blob::Ptr blob = GenerateInput(*info);
        float* rawBlobDataPtr = blob->buffer().as<float*>();
        const size_t blobSize = blob->size();
        float value = 0;
        for (size_t i = 0; i < blobSize; i++) {
            rawBlobDataPtr[i] = value;
            value = value + 1.f;
            if (value > 255.f) {
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
    bool enableLpt;
    std::tie(netPrecision, inputShape0, inputPrecision, enableLpt, targetDevice) = this->GetParam();

    const auto parameter = std::make_shared<ngraph::opset1::Parameter>(inputPrecision, ngraph::Shape{inputShape0});
    parameter->set_friendly_name("parameter");

    const auto convert1 = std::make_shared<ngraph::opset1::Convert>(parameter, ov::element::u8);
    convert1->set_friendly_name("convert1");

    const auto relu1 = std::make_shared<ngraph::opset1::Relu>(convert1);
    relu1->set_friendly_name("relu1");

    const auto convert2 = std::make_shared<ngraph::opset1::Convert>(relu1, ov::element::f32);
    convert2->set_friendly_name("convert2");

    const auto slope2 = std::make_shared<ngraph::opset1::Constant>(ov::element::f32, ov::Shape{}, std::vector<float>{-1.f});
    const auto relu2 = std::make_shared<ngraph::opset1::PRelu>(convert2, slope2);
    relu2->set_friendly_name("relu2");

    const auto min = std::make_shared<ngraph::opset1::Constant>(inputPrecision, ov::Shape{}, std::vector<float>{enableLpt ? 0.f : 1.f});
    const auto max = std::make_shared<ngraph::opset1::Constant>(inputPrecision, ov::Shape{}, std::vector<float>{ 25.5f });
    const auto fakeQuantize = std::make_shared<ngraph::opset1::FakeQuantize>(relu2, min, max, min, max, 256ul);
    fakeQuantize->set_friendly_name("fakeQuantize");

    const auto relu3 = std::make_shared<ngraph::opset1::Relu>(fakeQuantize);
    relu3->set_friendly_name("relu3");

    const auto result = std::make_shared<ngraph::opset1::Result>(relu3);
    result->set_friendly_name("result");

    function = std::make_shared<ngraph::Function>(ngraph::ResultVector{result}, ngraph::ParameterVector{parameter}, "CodegenQuantized");
}

void CodegenQuantized::Run() {
    LayerTestsCommon::Run();

    auto execGraph = getExecGraphInfoAsMap();
    const bool modelInLowPrecision = std::get<3>(this->GetParam());
    if (modelInLowPrecision) {
        EXPECT_EQ(12, execGraph.size());
        // four constants
        EXPECT_EQ("Input", execGraph.find("parameter")->second);
        EXPECT_EQ("Convert", execGraph.find("convert1")->second);
        EXPECT_EQ("Eltwise", execGraph.find("relu1")->second);
        EXPECT_EQ("Convert", execGraph.find("convert2")->second);
        EXPECT_EQ("Subgraph", execGraph.find("relu2")->second);
        EXPECT_EQ("FakeQuantize", execGraph.find("fakeQuantize")->second);
        EXPECT_EQ("Eltwise", execGraph.find("relu3_original,relu3")->second);
        EXPECT_EQ("Output", execGraph.find("result")->second);
    } else {
        EXPECT_EQ(6, execGraph.size());
        EXPECT_TRUE(execGraph.find("fakeQuantize") == execGraph.end());
        EXPECT_EQ("Input", execGraph.find("parameter")->second);
        EXPECT_EQ("Convert", execGraph.find("convert1")->second);
        EXPECT_EQ("Eltwise", execGraph.find("relu1")->second);
        EXPECT_EQ("Convert", execGraph.find("convert2")->second);
        EXPECT_EQ("Subgraph", execGraph.find("relu3")->second); // relu2 + fakeQuantize + relu3
        EXPECT_EQ("Output", execGraph.find("result")->second);
    }
}

TEST_P(CodegenQuantized, CompareWithRefImpl) {
    Run();
};

}  // namespace LayerTestsDefinitions
