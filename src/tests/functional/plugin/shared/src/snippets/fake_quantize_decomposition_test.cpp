
// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "subgraph_tests/fake_quantize_decomposition_test.hpp"

#include <memory>
#include <tuple>
#include <vector>
#include <string>

#include <ie_core.hpp>
#include "ngraph_ops/type_relaxed.hpp"
#include "fake_quantize_function.hpp"

namespace LayerTestsDefinitions {

std::string FakeQuantizeDecompositionTest::getTestCaseName(testing::TestParamInfo<TestValues> obj) {
    std::ostringstream result;
    result << "IS=" << CommonTestUtils::vec2str(obj.param.actual.inputShape) << "_";
    result << "netPRC=" << obj.param.actual.modelType << "_";
    result << "D=" << obj.param.actual.targetDevice << "_";
    result << "IN=" << obj.param.actual.inputType << "_";
    result << "LP=" << obj.param.actual.zeroPoint;
    result << "SH1=" << obj.param.actual.fakeQuantizeShapes[0] << "SH2=" << obj.param.actual.fakeQuantizeShapes[1]
           << "SH3=" << obj.param.actual.fakeQuantizeShapes[2] << "SH4=" << obj.param.actual.fakeQuantizeShapes[3];
    return result.str();
}

void FakeQuantizeDecompositionTest::GenerateInputs() {
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

void FakeQuantizeDecompositionTest::SetUp() {
    auto& values = this->GetParam();
    targetDevice = values.actual.targetDevice;

    function = ov::test::snippets::FakeQuantizeFunction::get(
        values.actual.inputShape,
        values.actual.inputType,
        values.actual.fakeQuantizeShapes,
        values.actual.zeroPoint);
}

void FakeQuantizeDecompositionTest::Run() {
    LayerTestsCommon::Run();

    auto& values = this->GetParam();

    auto execGraph = getExecGraphInfoAsMap();
    EXPECT_EQ(
        values.expected.operationsCount == -1 ? values.expected.expectedOperations.size() : values.expected.operationsCount,
        execGraph.size());

    for (const auto& operation : values.expected.expectedOperations) {
        const auto it = execGraph.find(operation.name);
        EXPECT_TRUE(it != execGraph.end()) << "operation '" << operation.name << "' was not found in result execution graph";
        if (it != execGraph.end()) {
            EXPECT_EQ(it->second, operation.type);
        }
    }

    for (const auto& notExpectedType : values.expected.notExpectedOperationTypes) {
        for (const auto it : execGraph) {
            const std::string type = it.second;
            EXPECT_NE(notExpectedType, type) << "not expected type '" << type << "' was found in execution graph";
        }
    }
}

TEST_P(FakeQuantizeDecompositionTest, CompareWithRefImpl) {
    Run();
};

}  // namespace LayerTestsDefinitions
