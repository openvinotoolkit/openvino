// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_layer_tests/memory.h"

#include <vector>

#include "ngraph_functions/builders.hpp"
#include "openvino/op/util/variable.hpp"

using namespace LayerTestsDefinitions;
using namespace ngraph;
using namespace opset7;

namespace {

class MemoryTestGna : public MemoryTest {
    using Super = MemoryTest;

protected:
    void SetUpTransformNone() override {
        auto param = builder::makeParams(ngPrc, {inputShape});
        const auto variable_info = targetDevice == CommonTestUtils::DEVICE_GPU
                                       ? VariableInfo{Shape{inputShape}, ngPrc, "v0"}
                                       : VariableInfo{PartialShape::dynamic(), element::dynamic, "v0"};
        auto variable = std::make_shared<Variable>(variable_info);

        auto min55 = ngraph::builder::makeConstant<float>(ngPrc, {}, {-55.001678466796875f});
        auto max55 = ngraph::builder::makeConstant<float>(ngPrc, {}, {55.0f});

        auto min270 = ngraph::builder::makeConstant<float>(ngPrc, {}, {-270.00823974609375f});
        auto max270 = ngraph::builder::makeConstant<float>(ngPrc, {}, {270.0f});

        auto min325 = ngraph::builder::makeConstant<float>(ngPrc, {}, {-325.0099182128906f});
        auto max325 = ngraph::builder::makeConstant<float>(ngPrc, {}, {325.0f});

        auto fq_from_Par225 = std::make_shared<opset9::FakeQuantize>(param.at(0), min55, max55, min55, max55, 65536);

        auto read_value = CreateReadValueOp(fq_from_Par225, variable);

        auto fq_from_ReadVal =
            std::make_shared<opset9::FakeQuantize>(read_value, min270, max270, min270, max270, 65536);

        auto add = std::make_shared<Add>(fq_from_ReadVal, fq_from_Par225);

        auto fq_from_add = std::make_shared<opset9::FakeQuantize>(add, min325, max325, min325, max325, 65536);

        auto assign = CreateAssignOp(fq_from_add, variable);
        auto res = std::make_shared<Result>(fq_from_add);

        function = std::make_shared<Function>(ResultVector{res}, SinkVector{assign}, param, "TestMemory");
    }
};

TEST_P(MemoryTestGna, CompareWithRefs) {
    Run();
}

std::vector<ngraph::helpers::MemoryTransformation> transformation{
    ngraph::helpers::MemoryTransformation::NONE,
    ngraph::helpers::MemoryTransformation::LOW_LATENCY_V2,
    ngraph::helpers::MemoryTransformation::LOW_LATENCY_V2_REGULAR_API,
    ngraph::helpers::MemoryTransformation::LOW_LATENCY_V2_ORIGINAL_INIT};

const std::vector<InferenceEngine::SizeVector> inShapes = {{1, 1}, {1, 2}, {1, 10}};

const std::vector<InferenceEngine::Precision> inputPrecisions = {InferenceEngine::Precision::FP32};

const std::vector<int64_t> iterationCount{1, 3, 4, 10};

INSTANTIATE_TEST_SUITE_P(smoke_MemoryTest,
                         MemoryTestGna,
                         ::testing::Combine(::testing::ValuesIn(transformation),
                                            ::testing::ValuesIn(iterationCount),
                                            ::testing::ValuesIn(inShapes),
                                            ::testing::ValuesIn(inputPrecisions),
                                            ::testing::Values(CommonTestUtils::DEVICE_GNA)),
                         MemoryTest::getTestCaseName);

}  // namespace
