// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <debug.h>
#include "functional_test_utils/precision_utils.hpp"
#include "functional_test_utils/skip_tests_config.hpp"
#include "subgraph_tests/scaleshift.hpp"

namespace LayerTestsDefinitions {
    std::string ScaleShiftLayerTest::getTestCaseName(const testing::TestParamInfo<ScaleShiftParamsTuple> &obj) {
        std::vector<std::vector<size_t>> inputShapes;
        InferenceEngine::Precision netPrecision;
        std::string targetName;
        std::vector<float> scale, shift;
        std::tie(inputShapes, netPrecision, targetName, scale, shift) = obj.param;
        std::ostringstream results;

        results << "IS=" << CommonTestUtils::vec2str(inputShapes) << "_";
        results << "Scale=" << CommonTestUtils::vec2str(scale) << "_";
        results << "Shift=" << CommonTestUtils::vec2str(shift) << "_";
        results << "netPRC=" << netPrecision.name() << "_";
        results << "targetDevice=" << targetName << "_";
        return results.str();
    }

    void ScaleShiftLayerTest::SetUp() {
        std::vector<std::vector<size_t>> inputShapes;
        InferenceEngine::Precision netPrecision;
        std::vector<float> scale, shift;
        std::tie(inputShapes, netPrecision, targetDevice, scale, shift) = this->GetParam();
        auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);
        auto paramsIn = ngraph::builder::makeParams(ngPrc, {inputShapes[0]});
        auto mul_const = std::make_shared<ngraph::op::Constant>(ngPrc, ngraph::Shape{1}, scale);
        auto mul = std::make_shared<ngraph::opset1::Multiply>(paramsIn[0], mul_const);
        auto add_const = std::make_shared<ngraph::op::Constant>(ngPrc, ngraph::Shape{1}, shift);
        auto add = std::make_shared<ngraph::opset1::Add>(mul, add_const);
        function = std::make_shared<ngraph::Function>(add, paramsIn, "scale_shift");
    }

    TEST_P(ScaleShiftLayerTest, CompareWithRefs){
        Run();
    };
} // namespace LayerTestsDefinitions
