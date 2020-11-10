// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <memory>
#include <tuple>
#include <vector>
#include <string>

#include <ie_core.hpp>

#include "common_test_utils/common_utils.hpp"
#include "functional_test_utils/plugin_cache.hpp"
#include "functional_test_utils/layer_test_utils.hpp"
#include "functional_test_utils/blob_utils.hpp"

#include "ngraph_functions/pass/convert_prc.hpp"

#include "subgraph_tests/first_connect_input_concat.hpp"


namespace LayerTestsDefinitions {

std::string ConcatFirstInputTest::getTestCaseName(testing::TestParamInfo<concatFirstInputParams> obj) {
    std::vector<std::vector<size_t>> inputShapes;
    InferenceEngine::Precision netPrecision;
    std::string targetDevice;
    std::map<std::string, std::string> additional_config;
    std::tie(inputShapes, netPrecision, targetDevice, additional_config) = obj.param;

    std::ostringstream result;
    result << "IS=" << CommonTestUtils::vec2str(inputShapes) << "_";
    result << "netPRC=" << netPrecision.name() << "_";
    result << "targetDevice=" << targetDevice;

    return result.str();
}

void ConcatFirstInputTest::SetUp() {
    std::vector<std::vector<size_t>> inputShapes;
    InferenceEngine::Precision netPrecision;
    std::map<std::string, std::string> additional_config;
    std::tie(inputShapes, netPrecision, targetDevice, configuration) = this->GetParam();
    auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);
    auto params = ngraph::builder::makeParams(ngPrc, inputShapes);
    auto const_second_param = ngraph::builder::makeConstant(ngPrc, {1, 8}, std::vector<float>{-1.0f});
    auto concat = std::make_shared<ngraph::opset1::Concat>(ngraph::OutputVector{params[0], const_second_param}, 1);
    auto relu = std::make_shared<ngraph::opset1::Relu>(concat);

    ngraph::ResultVector results{std::make_shared<ngraph::opset1::Result>(relu)};

    function = std::make_shared<ngraph::Function>(results, params, "ConcatMultiInput");
}

TEST_P(ConcatFirstInputTest, CompareWithRefImpl) {
    Run();
};
}  // namespace LayerTestsDefinitions
