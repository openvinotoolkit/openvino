// Copyright (C) 2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <tuple>
#include <string>
#include <vector>
#include <memory>
#include <functional>

#include "ie_core.hpp"

#include "common_test_utils/common_utils.hpp"
#include "functional_test_utils/blob_utils.hpp"
#include "common_test_utils/data_utils.hpp"
#include "functional_test_utils/precision_utils.hpp"
#include "functional_test_utils/plugin_cache.hpp"
#include "functional_test_utils/skip_tests_config.hpp"

#include "single_layer_tests/concat_4D.hpp"

namespace LayerTestsDefinitions {

    std::string Concat4DLayerTest::getTestCaseName(const testing::TestParamInfo<concat4DParamsTuple> &obj) {
        int axis;
        std::vector<size_t> inputShapes;
        InferenceEngine::Precision netPrecision;
        InferenceEngine::Precision inPrc, outPrc;
        InferenceEngine::Layout inLayout, outLayout;
        std::string targetName;
        std::map<std::string, std::string> config;
        std::tie(inputShapes, netPrecision, targetName, config) = obj.param;
        std::ostringstream result;
        result << "IS=" << CommonTestUtils::vec2str(inputShapes) << "_";
        result << "netPRC=" << netPrecision.name() << "_";
        result << "trgDev=" << targetName << "_";
        return result.str();
    }

    void Concat4DLayerTest::SetUp() {
        int axis = 1;
        InferenceEngine::SizeVector inputShape;
        InferenceEngine::Precision netPrecision;
        std::map<std::string, std::string> additional_config;
        std::tie(inputShape, netPrecision, targetDevice, additional_config) = this->GetParam();
        configuration.insert(additional_config.begin(), additional_config.end());

        auto total_size = std::accumulate(inputShape.begin(), inputShape.end(), static_cast<size_t>(1), std::multiplies<size_t>());
        auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);
        auto params = ngraph::builder::makeParams(ngPrc, {inputShape});
        auto input = params[0];

        auto constant_values = CommonTestUtils::generate_float_numbers(total_size, 11.0f, 12.0f);
        auto constant = ngraph::builder::makeConstant(ngPrc, std::vector<size_t>({1, total_size}), constant_values);
        auto first_reshape_pattern = std::make_shared<ngraph::op::Constant>(ngraph::element::i64,
            ngraph::Shape{4}, std::vector<size_t>(inputShape));
        auto first_reshape = std::make_shared<ngraph::op::v1::Reshape>(constant, first_reshape_pattern, false);
        auto constant_2 = ngraph::builder::makeConstant(ngPrc, inputShape, constant_values);

        auto concat = std::make_shared<ngraph::opset1::Concat>(ngraph::OutputVector({first_reshape, input, constant_2}), axis);
        auto act = ngraph::builder::makeActivation(concat, ngPrc, ngraph::helpers::ActivationTypes::Relu);
        ngraph::ResultVector results{std::make_shared<ngraph::opset1::Result>(act)};
        function = std::make_shared<ngraph::Function>(results, params, "concat");
    }


    TEST_P(Concat4DLayerTest, CompareWithRefs) {
        Run();
    };
}  // namespace LayerTestsDefinitions
