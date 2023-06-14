// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "concat.hpp"

#include "gtest/gtest.h"
#include "test_utils/cpu_test_utils.hpp"

using namespace InferenceEngine;
using namespace CPUTestUtils;
using namespace ngraph::helpers;
using namespace ov::test;

namespace CPULayerTestsDefinitions {
std::string ConcatLayerCPUTest::getTestCaseName(testing::TestParamInfo<concatCPUTestParams> obj) {
    int axis;
    std::vector<InputShape> inputShapes;
    ElementType netPrecision;
    CPUSpecificParams cpuParams;
    std::tie(axis, inputShapes, netPrecision, cpuParams) = obj.param;

    std::ostringstream result;
    result << "IS=";
    for (const auto& shape : inputShapes) {
        result << CommonTestUtils::partialShape2str({shape.first}) << "_";
    }
    result << "TS=";
    for (const auto& shape : inputShapes) {
        result << "(";
        if (!shape.second.empty()) {
            for (const auto& itr : shape.second) {
                result << CommonTestUtils::vec2str(itr);
            }
        }
        result << ")_";
    }
    result << "axis=" << axis << "_";
    result << "netPRC=" << netPrecision << "_";
    result << CPUTestsBase::getTestCaseName(cpuParams);
    return result.str();
}

void ConcatLayerCPUTest::compare(const std::vector<ov::Tensor>& expected, const std::vector<ov::Tensor>& actual) {
    if (actual.front().get_size() == 0) {
        ASSERT_EQ(0, expected.front().get_size());
        for (const auto& shape : targetStaticShapes[inferNum]) {
            ASSERT_EQ(shape_size(shape), 0);
        }
    } else {
        SubgraphBaseTest::compare(expected, actual);
    }
    inferNum++;
}

void ConcatLayerCPUTest::SetUp() {
    targetDevice = CommonTestUtils::DEVICE_CPU;

    int axis;
    std::vector<InputShape> inputShape;
    ElementType netPrecision;
    CPUSpecificParams cpuParams;
    std::tie(axis, inputShape, netPrecision, cpuParams) = this->GetParam();

    std::tie(inFmts, outFmts, priority, selectedType) = cpuParams;
    selectedType += std::string("_") + InferenceEngine::details::convertPrecision(netPrecision).name();

    init_input_shapes(inputShape);

    auto params = ngraph::builder::makeDynamicParams(netPrecision, inputDynamicShapes);
    auto paramOuts =
        ngraph::helpers::convert2OutputVector(ngraph::helpers::castOps2Nodes<ngraph::op::Parameter>(params));
    auto concat = std::make_shared<ngraph::opset1::Concat>(paramOuts, axis);

    function = makeNgraphFunction(netPrecision, params, concat, "ConcatCPU");
}

TEST_P(ConcatLayerCPUTest, CompareWithRefs) {
    run();
    CheckPluginRelatedResults(compiledModel, "Concatenation");
}

namespace Concat {

}  // namespace Concat
}  // namespace CPULayerTestsDefinitions