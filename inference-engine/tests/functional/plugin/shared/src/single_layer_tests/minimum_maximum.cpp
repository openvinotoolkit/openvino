// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <tuple>
#include <string>
#include <vector>
#include <memory>
#include <debug.h>

#include "common_test_utils/common_utils.hpp"
#include "functional_test_utils/precision_utils.hpp"
#include "functional_test_utils/skip_tests_config.hpp"
#include "single_layer_tests/minimum_maximum.hpp"

namespace LayerTestsDefinitions {
    std::string MaxMinLayerTest::getTestCaseName(const testing::TestParamInfo<MaxMinParamsTuple> &obj) {
        std::vector<std::vector<size_t>> inputShapes;
        InferenceEngine::Precision netPrecision;
        std::string targetName;
        ngraph::helpers::InputLayerType inputType;
        ngraph::helpers::MinMaxOpType opType;
        std::tie(inputShapes, opType, netPrecision, inputType, targetName) = obj.param;
        std::ostringstream results;

        results << "IS=" << CommonTestUtils::vec2str(inputShapes) << "_";
        results << "OpType=" << opType << "_";
        results << "SecondaryInputType=" << inputType << "_";
        results << "netPRC=" << netPrecision.name() << "_";
        results << "targetDevice=" << targetName << "_";
        return results.str();
    }

    void MaxMinLayerTest::SetUp() {
        std::vector<std::vector<size_t>> inputShapes;
        InferenceEngine::Precision netPrecision;
        ngraph::helpers::InputLayerType inputType;
        ngraph::helpers::MinMaxOpType opType;
        std::tie(inputShapes, opType, netPrecision, inputType, targetDevice) = this->GetParam();
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

    TEST_P(MaxMinLayerTest, CompareWithRefs){
        Run();
    };
} // namespace LayerTestsDefinitions
