
// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <memory>
#include <tuple>
#include <vector>
#include <string>

#include <ie_core.hpp>
#include "subgraph_tests/codegen_floor.hpp"

namespace LayerTestsDefinitions {

    std::string CodegenFloor::getTestCaseName(testing::TestParamInfo<LayerTestsDefinitions::inputParams> obj) {
        InferenceEngine::Precision netPrecision;
        InferenceEngine::SizeVector inputShapes, newInputShapes;
        std::string targetDevice;
        std::tie(netPrecision, inputShapes, targetDevice) = obj.param;

        std::ostringstream result;
        result << "IS=" << CommonTestUtils::vec2str(inputShapes) << "_";
        result << "netPRC=" << netPrecision.name() << "_";
        result << "targetDevice=" << targetDevice;
        return result.str();
    }

    void CodegenFloor::GenerateInputs() {
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
            for (size_t i = 0; i < blobSize; i++) {
                rawBlobDataPtr[i] = static_cast<float>(i) + 0.9f;
            }
            inputs.push_back(blob);
        }
    }

    void CodegenFloor::SetUp() {
        std::vector<size_t> inputShape0;
        InferenceEngine::Precision netPrecision;
        std::tie(netPrecision, inputShape0, targetDevice) = this->GetParam();

        auto input = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape{inputShape0});

        auto floor = std::make_shared<ngraph::opset1::Floor>(input);
        auto result = std::make_shared<ngraph::opset1::Result>(floor);

        function = std::make_shared<ngraph::Function>(
            ngraph::ResultVector{result},
            ngraph::ParameterVector{input},
            "CodegenFloor");
    }

TEST_P(CodegenFloor, CompareWithRefImpl) {
    Run();
};

}  // namespace LayerTestsDefinitions
