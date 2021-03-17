// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/single_layer/minimum_maximum.hpp"

namespace LayerTestsDefinitions {
    std::string MaxMinLayerTest::getTestCaseName(const testing::TestParamInfo<MaxMinParamsTuple> &obj) {
        std::vector<std::vector<size_t>> inputShapes;
        InferenceEngine::Precision netPrecision;
        InferenceEngine::Precision inPrc, outPrc;
        InferenceEngine::Layout inLayout, outLayout;
        std::string targetName;
        ngraph::helpers::InputLayerType inputType;
        ngraph::helpers::MinMaxOpType opType;
        std::tie(inputShapes, opType, netPrecision, inPrc, outPrc, inLayout, outLayout, inputType, targetName) = obj.param;
        std::ostringstream results;

        results << "IS=" << CommonTestUtils::vec2str(inputShapes) << "_";
        results << "OpType=" << opType << "_";
        results << "SecondaryInputType=" << inputType << "_";
        results << "netPRC=" << netPrecision.name() << "_";
        results << "inPRC=" << inPrc.name() << "_";
        results << "outPRC=" << outPrc.name() << "_";
        results << "inL=" << inLayout << "_";
        results << "outL=" << outLayout << "_";
        results << "trgDev=" << targetName << "_";
        return results.str();
    }

    void MaxMinLayerTest::SetUp() {
        std::vector<std::vector<size_t>> inputShapes;
        InferenceEngine::Precision netPrecision;
        ngraph::helpers::InputLayerType inputType;
        ngraph::helpers::MinMaxOpType opType;
        std::tie(inputShapes, opType, netPrecision, inPrc, outPrc, inLayout, outLayout, inputType, targetDevice) = this->GetParam();
        if (inputShapes.size() != 2) {
            THROW_IE_EXCEPTION << "Unsupported inputs number for Minimum/Maximum operaton";
        }
        auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);
        auto input = ngraph::builder::makeParams(ngPrc, {inputShapes[0]});
        auto secondaryInput = ngraph::builder::makeInputLayer(ngPrc, inputType, {inputShapes[1]});
        if (inputType == ngraph::helpers::InputLayerType::PARAMETER) {
            input.push_back(std::dynamic_pointer_cast<ngraph::opset3::Parameter>(secondaryInput));
        }

        auto op = ngraph::builder::makeMinMax(input[0], secondaryInput, opType);
        function = std::make_shared<ngraph::Function>(op, input, "MinMax");
    }
} // namespace LayerTestsDefinitions
