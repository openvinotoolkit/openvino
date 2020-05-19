// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <tuple>
#include <string>
#include <vector>
#include <functional>

#include <ie_core.hpp>
#include <ie_precision.hpp>

#include "functional_test_utils/blob_utils.hpp"
#include "functional_test_utils/precision_utils.hpp"
#include "common_test_utils/common_utils.hpp"
#include "functional_test_utils/skip_tests_config.hpp"
#include "functional_test_utils/plugin_cache.hpp"

#include "single_layer_tests/select.hpp"

namespace LayerTestsDefinitions {
    enum { CONDITION, THEN, ELSE, numOfInputs };

    std::string SelectLayerTest::getTestCaseName(const testing::TestParamInfo<selectTestParams> &obj) {
        std::vector<std::vector<size_t>> dataShapes(3);
        InferenceEngine::Precision dataType;
        ngraph::op::AutoBroadcastSpec broadcast;
        std::string targetDevice;
        std::tie(dataShapes, dataType, broadcast, targetDevice) = obj.param;
        std::ostringstream result;
        result << "COND=BOOL_" << CommonTestUtils::vec2str(dataShapes[CONDITION]);
        result << "_THEN=" << dataType.name() << "_" << CommonTestUtils::vec2str(dataShapes[THEN]);
        result << "_ELSE=" << dataType.name() << "_" << CommonTestUtils::vec2str(dataShapes[ELSE]);
        result << "_" << broadcast.m_type;
        result << "_targetDevice=" << targetDevice;
        return result.str();
    }

    void SelectLayerTest::SetUp() {
        SetRefMode(LayerTestsUtils::RefMode::CONSTANT_FOLDING);

        std::vector<std::vector<size_t>> inputShapes(numOfInputs);
        InferenceEngine::Precision inputPrecision;
        ngraph::op::AutoBroadcastSpec broadcast;
        std::tie(inputShapes, inputPrecision, broadcast, targetDevice) = this->GetParam();

        ngraph::ParameterVector paramNodesVector;
        auto paramNode = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::Type_t::boolean, ngraph::Shape(inputShapes[CONDITION]));
        paramNodesVector.push_back(paramNode);
        auto inType = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(inputPrecision);
        for (size_t i = 1; i < inputShapes.size(); i++) {
            paramNode = std::make_shared<ngraph::opset1::Parameter>(inType, ngraph::Shape(inputShapes[i]));
            paramNodesVector.push_back(paramNode);
        }
        auto paramOuts = ngraph::helpers::convert2OutputVector(ngraph::helpers::castOps2Nodes<ngraph::op::Parameter>(paramNodesVector));

        auto select = std::dynamic_pointer_cast<ngraph::opset1::Select>(ngraph::builder::makeSelect(paramOuts, broadcast));
        ngraph::ResultVector results{std::make_shared<ngraph::opset1::Result>(select)};
        function = std::make_shared<ngraph::Function>(results, paramNodesVector, "select");
    }

    TEST_P(SelectLayerTest, CompareWithRefImpl) {
        Run();

        if (targetDevice == std::string{CommonTestUtils::DEVICE_GPU}) {
            PluginCache::get().reset();
        }
    }

}  // namespace LayerTestsDefinitions
