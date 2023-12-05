// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/single_layer/power.hpp"

namespace LayerTestsDefinitions {
    std::string PowerLayerTest::getTestCaseName(const testing::TestParamInfo<PowerParamsTuple> &obj) {
        std::vector<std::vector<size_t>> inputShapes;
        InferenceEngine::Precision netPrecision;
        InferenceEngine::Precision inPrc, outPrc;
        InferenceEngine::Layout inLayout, outLayout;
        std::string targetName;
        std::vector<float> power;
        std::tie(inputShapes, netPrecision, inPrc, outPrc, inLayout, outLayout, targetName, power) = obj.param;
        std::ostringstream results;

        results << "IS=" << ov::test::utils::vec2str(inputShapes) << "_";
        results << "Power=" << ov::test::utils::vec2str(power) << "_";
        results << "netPRC=" << netPrecision.name() << "_";
        results << "inPRC=" << inPrc.name() << "_";
        results << "outPRC=" << outPrc.name() << "_";
        results << "inL=" << inLayout << "_";
        results << "outL=" << outLayout << "_";
        results << "trgDev=" << targetName << "_";
        return results.str();
    }

    void PowerLayerTest::SetUp() {
        threshold = 0.04f;

        std::vector<std::vector<size_t>> inputShapes;
        InferenceEngine::Precision netPrecision;
        std::vector<float> power;
        std::tie(inputShapes, netPrecision, inPrc, outPrc, inLayout, outLayout, targetDevice, power) = this->GetParam();
        auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);
        ov::ParameterVector paramsIn{std::make_shared<ov::op::v0::Parameter>(ngPrc, ov::Shape(inputShapes[0]))};

        auto power_const = std::make_shared<ngraph::op::Constant>(ngPrc, ngraph::Shape{ 1 }, power);
        auto pow = std::make_shared<ngraph::opset1::Power>(paramsIn[0], power_const);

        function = std::make_shared<ngraph::Function>(pow, paramsIn, "power");
    }
} // namespace LayerTestsDefinitions
