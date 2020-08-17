// Copyright (C) 2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <tuple>
#include <string>
#include <vector>
#include <memory>
#include <functional>
#include <functional_test_utils/skip_tests_config.hpp>

#include "ie_core.hpp"
#include "ngraph_functions/utils/ngraph_helpers.hpp"

#include "common_test_utils/common_utils.hpp"
#include "functional_test_utils/blob_utils.hpp"
#include "functional_test_utils/plugin_cache.hpp"
#include "functional_test_utils/layer_test_utils.hpp"

#include "single_layer_tests/variadic_split.hpp"

namespace LayerTestsDefinitions {

    std::string VariadicSplitLayerTest::getTestCaseName(testing::TestParamInfo<VariadicSplitParams> obj) {
        size_t axis;
        std::vector<size_t> numSplits;
        InferenceEngine::Precision netPrecision;
        InferenceEngine::SizeVector inputShapes;
        std::string targetDevice;
        std::tie(numSplits, axis, netPrecision, inputShapes, targetDevice) = obj.param;
        std::ostringstream result;
        result << "IS=" << CommonTestUtils::vec2str(inputShapes) << "_";
        result << "numSplits=" << CommonTestUtils::vec2str(numSplits) << "_";
        result << "axis=" << axis << "_";
        result << "IS";
        result << "netPRC=" << netPrecision.name() << "_";
        result << "targetDevice=" << targetDevice;
        return result.str();
    }

    void VariadicSplitLayerTest::SetUp() {
        SetRefMode(LayerTestsUtils::RefMode::CONSTANT_FOLDING);
        size_t axis;
        std::vector<size_t> inputShape, numSplits;
        InferenceEngine::Precision netPrecision;
        std::tie(numSplits, axis, netPrecision, inputShape, targetDevice) = this->GetParam();
        auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);
        auto params = ngraph::builder::makeParams(ngPrc, {inputShape});
        auto paramOuts = ngraph::helpers::convert2OutputVector(
                ngraph::helpers::castOps2Nodes<ngraph::opset3::Parameter>(params));
        auto VariadicSplit = std::dynamic_pointer_cast<ngraph::opset3::VariadicSplit>(ngraph::builder::makeVariadicSplit(params[0], numSplits, axis));
        ngraph::ResultVector results;
        for (int i = 0; i < numSplits.size(); i++) {
            results.push_back(std::make_shared<ngraph::opset3::Result>(VariadicSplit->output(i)));
        }
        function = std::make_shared<ngraph::Function>(results, params, "VariadicSplit");
    }

    TEST_P(VariadicSplitLayerTest, CompareWithRefs) {
        Run();
    }

}  // namespace LayerTestsDefinitions