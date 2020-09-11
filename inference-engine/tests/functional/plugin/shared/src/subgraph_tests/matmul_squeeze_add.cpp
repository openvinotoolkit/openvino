// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <ie_core.hpp>
#include <memory>
#include <string>
#include <tuple>
#include <vector>
#include <ie_plugin_config.hpp>

#include "common_test_utils/common_utils.hpp"
#include "functional_test_utils/blob_utils.hpp"
#include "functional_test_utils/layer_test_utils.hpp"
#include "functional_test_utils/plugin_cache.hpp"
#include "ngraph_functions/pass/convert_prc.hpp"

#include "subgraph_tests/matmul_squeeze_add.hpp"

#include "ngraph_functions/builders.hpp"

namespace LayerTestsDefinitions {

std::string MatmulSqueezeAddTest::getTestCaseName(testing::TestParamInfo<matmulSqueezeAddParams> obj) {
    InferenceEngine::Precision netPrecision;
    std::vector<size_t> inputShape;
    std::size_t outputSize;
    std::string targetDevice;
    std::map<std::string, std::string> configuration;
    std::tie(netPrecision, targetDevice, configuration, inputShape, outputSize) = obj.param;

    std::ostringstream result;
    result << "IS=" << CommonTestUtils::vec2str(inputShape) << "_";
    result << "OS=" << outputSize << "_";
    result << "netPRC=" << netPrecision.name() << "_";
    result << "targetDevice=" << targetDevice;
    for (auto const& configItem : configuration) {
        result << "_configItem=" << configItem.first << "_" << configItem.second;
    }
    return result.str();
}

void MatmulSqueezeAddTest::SetUp() {
    auto generateFloatNumbers = [](float startFrom, float upTo, std::size_t vec_len) {
        std::vector<float> res;

        std::mt19937 gen(
            static_cast<float>(std::chrono::high_resolution_clock::now().time_since_epoch().count()));

        std::uniform_real_distribution<float> dist(startFrom, upTo);

        for (int i = 0; i < vec_len; i++)
            res.emplace_back(static_cast<float>(dist(gen)));

        return res;
    };

    InferenceEngine::Precision netPrecision;
    std::map<std::string, std::string> tempConfig;
    std::vector<size_t> inputShape;
    size_t outputSize;
    std::tie(netPrecision, targetDevice, tempConfig, inputShape, outputSize) = this->GetParam();
    configuration.insert(tempConfig.begin(), tempConfig.end());

    auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);

    auto params = ngraph::builder::makeParams(ngPrc, { inputShape });

    auto constant_0 = ngraph::builder::makeConstant<float>(ngPrc, { outputSize, inputShape[1] },
        generateFloatNumbers(0, 1, outputSize * inputShape[1]), false);
    auto matmul_0 = std::make_shared<ngraph::op::MatMul>(params[0], constant_0, false, true);

    auto constant_1 = std::make_shared<ngraph::op::Constant>(ngraph::element::Type_t::i64, ngraph::Shape{ 1 }, std::vector<size_t>{0});
    auto unsqueeze_0 = std::make_shared<ngraph::op::Unsqueeze>(matmul_0, constant_1);

    auto constant_2 = ngraph::builder::makeConstant<float>(ngPrc, { 1, inputShape[0], outputSize },
        generateFloatNumbers(0, 1, inputShape[0] * outputSize), false);
    auto add_0 = std::make_shared<ngraph::op::Add>(unsqueeze_0, constant_2);

    auto constant_3 = std::make_shared<ngraph::op::Constant>(ngraph::element::Type_t::i64, ngraph::Shape{ 1 }, std::vector<size_t>{0});
    auto squeeze_0 = std::make_shared<ngraph::op::Squeeze>(add_0, constant_3);

    ngraph::ResultVector results {std::make_shared<ngraph::op::Result>(squeeze_0)};
    function = std::make_shared<ngraph::Function>(results, params, "MatmulSqueezeAddTest");
}

TEST_P(MatmulSqueezeAddTest, CompareWithRefImpl) {
    Run();
};

}  // namespace LayerTestsDefinitions
