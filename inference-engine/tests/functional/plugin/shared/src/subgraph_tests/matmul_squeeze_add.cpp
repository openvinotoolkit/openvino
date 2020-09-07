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
    InferenceEngine::SizeVector inputShapes, newInputShapes;
    std::string targetDevice;
    std::map<std::string, std::string> configuration;
    std::tie(netPrecision, targetDevice, configuration) = obj.param;

    std::ostringstream result;
    result << "IS=" << CommonTestUtils::vec2str(inputShapes) << "_";
    result << "netPRC=" << netPrecision.name() << "_";
    result << "targetDevice=" << targetDevice;
    for (auto const& configItem : configuration) {
        result << "_configItem=" << configItem.first << "_" << configItem.second;
    }
    return result.str();
}

void MatmulSqueezeAddTest::SetUp() {
    InferenceEngine::Precision netPrecision;
    std::tie(netPrecision, targetDevice, configuration) = this->GetParam();
    auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);

    auto params = ngraph::builder::makeParams(ngPrc, { {1, 512} });
    outPrc = InferenceEngine::Precision::FP32;

    auto constant_0 = ngraph::builder::makeConstant<float>(ngPrc, { 1000, 512 }, {}, true);
    auto matmul_0 = ngraph::builder::makeMatMul(params[0], constant_0, false, true);

    auto constant_1 = std::make_shared<ngraph::opset1::Constant>(ngraph::element::Type_t::i64, ngraph::Shape{ 1 }, std::vector<size_t>{0});
    auto squeeze_0 = std::make_shared<ngraph::opset1::Squeeze>(matmul_0, constant_1);

    auto constant_2 = ngraph::builder::makeConstant<float>(ngPrc, { 1, 1000 }, {}, true);
    auto add_0 = std::make_shared<ngraph::opset1::Add>(squeeze_0, constant_2);

    ngraph::ResultVector results {std::make_shared<ngraph::opset1::Result>(add_0)};
    function = std::make_shared<ngraph::Function>(results, params, "MatmulSqueezeAddTest");
}

TEST_P(MatmulSqueezeAddTest, CompareWithRefImpl) {
    Run();
};

}  // namespace LayerTestsDefinitions
