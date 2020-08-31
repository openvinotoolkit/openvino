// Copyright (C) 2019 Intel Corporation
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

#include "subgraph_tests/concat_multi_input.hpp"


namespace LayerTestsDefinitions {


std::string ConcatMultiInput::getTestCaseName(testing::TestParamInfo<concatQuantizationParams> obj) {
    size_t inputNum;
    std::vector<size_t> inputShapes;
    InferenceEngine::Precision netPrecision;
    std::string targetDevice;
    std::map<std::string, std::string> additional_config;
    std::tie(inputNum, inputShapes, netPrecision, targetDevice, additional_config) = obj.param;

    std::ostringstream result;
    result << "IN=" << inputNum << "_";
    result << "IS=" << CommonTestUtils::vec2str(inputShapes) << "_";
    result << "netPRC=" << netPrecision.name() << "_";
    result << "targetDevice=" << targetDevice;

    return result.str();
}

void ConcatMultiInput::SetUp() {
    size_t inputNum = 0;
    std::vector<size_t> inputShapes;
    InferenceEngine::Precision netPrecision;
    std::map<std::string, std::string> additional_config;
    std::tie(inputNum, inputShapes, netPrecision, targetDevice, additional_config) = this->GetParam();
    configuration.insert(additional_config.begin(), additional_config.end());

    auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);
    std::vector<size_t> paramsSize = { 1, inputShapes[1] * inputNum };
    auto params = ngraph::builder::makeParams(ngPrc, { paramsSize });
    auto stride = std::make_shared<ngraph::op::Constant>(ngraph::element::i64, ngraph::Shape{ 2 }, std::vector<int64_t>{ 1, 1 });

    std::vector<int64_t> newAxis = { 0, 0 };
    std::vector<int64_t> begin_mask = { 0, 0 };
    std::vector<int64_t> end_mask = { 0, 0 };
    std::vector<std::shared_ptr<ngraph::opset1::StridedSlice>> ssArray;
    ngraph::OutputVector concatInput;

    auto relu = std::make_shared<ngraph::opset1::Relu>(params[0]);
    for (int64_t i = 0; i < inputNum; ++i) {
        auto begin = std::make_shared<ngraph::op::Constant>(ngraph::element::i64, ngraph::Shape{ 2 },
            std::vector<int64_t>{ 0, i * static_cast<int64_t>(inputShapes[1]) });
        auto end = std::make_shared<ngraph::op::Constant>(ngraph::element::i64, ngraph::Shape{ 2 },
            std::vector<int64_t>{ 1, (i + 1) * static_cast<int64_t>(inputShapes[1]) });
        auto ss = std::make_shared<ngraph::opset1::StridedSlice>(relu, begin, end, stride, begin_mask, end_mask, newAxis);
        ssArray.push_back(ss);
        concatInput.push_back(ssArray[i]);
    }

    auto concat = std::make_shared<ngraph::opset1::Concat>(concatInput, 1);

    ngraph::ResultVector results{ std::make_shared<ngraph::opset1::Result>(concat) };
    function = std::make_shared<ngraph::Function>(results, params, "ConcatMultiInput");
}

TEST_P(ConcatMultiInput, CompareWithRefImpl) {
    Run();
};


}  // namespace LayerTestsDefinitions
