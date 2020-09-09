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

#include "subgraph_tests/concat_multi_input.hpp"


namespace LayerTestsDefinitions {


std::string ConcatMultiInput::getTestCaseName(testing::TestParamInfo<concatMultiParams> obj) {
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

void ConcatMultiInput::SetUp() {
    std::vector<std::vector<size_t>> inputShapes;
    InferenceEngine::Precision netPrecision;
    std::map<std::string, std::string> additional_config;
    std::tie(inputShapes, netPrecision, targetDevice, additional_config) = this->GetParam();
    configuration.insert(additional_config.begin(), additional_config.end());

    auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);
    std::vector<size_t> paramSize = { 1, 0 };
    for (const auto& val : inputShapes) {
        paramSize[1] += val[1];
    }
    auto params = ngraph::builder::makeParams(ngPrc, { paramSize });
    auto stride = std::make_shared<ngraph::op::Constant>(ngraph::element::i64, ngraph::Shape{ 2 }, std::vector<int64_t>{ 1, 1 });

    std::vector<int64_t> newAxis = { 0, 0 };
    std::vector<int64_t> begin_mask = { 0, 0 };
    std::vector<int64_t> end_mask = { 0, 0 };
    std::vector<std::shared_ptr<ngraph::opset1::StridedSlice>> ssArray;
    ngraph::OutputVector concatInput;

    auto relu = std::make_shared<ngraph::opset1::Relu>(params[0]);
    std::vector<int64_t> startOffset = { 0, 0 };
    for (size_t i = 0; i < inputShapes.size(); ++i) {
        std::vector<int64_t> shape = { static_cast<int64_t>(inputShapes[i][0]),
                                       static_cast<int64_t>(inputShapes[i][1]) };
        std::vector<int64_t> endoffset = { static_cast<int64_t>(inputShapes[i][0]) + startOffset[0],
                                           static_cast<int64_t>(inputShapes[i][1]) + startOffset[1]};
        auto begin = std::make_shared<ngraph::op::Constant>(ngraph::element::i64, ngraph::Shape{ 2 }, startOffset);
        auto end = std::make_shared<ngraph::op::Constant>(ngraph::element::i64, ngraph::Shape{ 2 }, endoffset);
        auto ss = std::make_shared<ngraph::opset1::StridedSlice>(relu, begin, end, stride, begin_mask, end_mask, newAxis);
        ssArray.push_back(ss);
        concatInput.push_back(ssArray[i]);

        startOffset[1] += shape[1];
    }

    auto concat = std::make_shared<ngraph::opset1::Concat>(concatInput, 1);

    ngraph::ResultVector results{ std::make_shared<ngraph::opset1::Result>(concat) };
    function = std::make_shared<ngraph::Function>(results, params, "ConcatMultiInput");
}

TEST_P(ConcatMultiInput, CompareWithRefImpl) {
    Run();
};


}  // namespace LayerTestsDefinitions
