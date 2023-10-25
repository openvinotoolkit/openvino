// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ov_models/builders.hpp"
#include "shared_test_classes/subgraph/scaleshift.hpp"

namespace SubgraphTestsDefinitions {
    std::string ScaleShiftLayerTest::getTestCaseName(const testing::TestParamInfo<ScaleShiftParamsTuple> &obj) {
        std::vector<std::vector<size_t>> inputShapes;
        InferenceEngine::Precision netPrecision;
        std::string targetName;
        std::vector<float> scale, shift;
        std::tie(inputShapes, netPrecision, targetName, scale, shift) = obj.param;
        std::ostringstream results;

        results << "IS=" << ov::test::utils::vec2str(inputShapes) << "_";
        results << "Scale=" << ov::test::utils::vec2str(scale) << "_";
        results << "Shift=" << ov::test::utils::vec2str(shift) << "_";
        results << "netPRC=" << netPrecision.name() << "_";
        results << "targetDevice=" << targetName << "_";
        return results.str();
    }

    void ScaleShiftLayerTest::SetUp() {
        std::vector<std::vector<size_t>> inputShapes;
        InferenceEngine::Precision netPrecision;
        std::vector<float> scale, shift;
        std::tie(inputShapes, netPrecision, targetDevice, scale, shift) = this->GetParam();
        auto paramsShape = ngraph::Shape{1};
        if (inputShapes.size() > 1)
            paramsShape = ngraph::Shape(inputShapes[1]);
        auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);
        ov::ParameterVector paramsIn{std::make_shared<ov::op::v0::Parameter>(ngPrc, ov::Shape(inputShapes[0]))};
        auto mul_const = std::make_shared<ngraph::op::Constant>(ngPrc, paramsShape, scale);
        auto mul = std::make_shared<ngraph::opset1::Multiply>(paramsIn[0], mul_const);
        auto add_const = std::make_shared<ngraph::op::Constant>(ngPrc, paramsShape, shift);
        auto add = std::make_shared<ngraph::opset1::Add>(mul, add_const);
        function = std::make_shared<ngraph::Function>(add, paramsIn, "scale_shift");
    }
} // namespace SubgraphTestsDefinitions
