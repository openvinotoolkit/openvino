// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/subgraph/connect_split_concat_concat.hpp"

namespace SubgraphTestsDefinitions {
std::string SplitConcatConcatTest::getTestCaseName(const testing::TestParamInfo<SplitConcatConcatParams> &obj) {
    InferenceEngine::Precision netPrecision;
    std::string targetDevice;
    std::map<std::string, std::string> configuration;
    std::tie(netPrecision, targetDevice, configuration) = obj.param;

    std::ostringstream result;
    result << "netPRC=" << netPrecision.name() << "_";
    result << "targetDevice=" << targetDevice << "_";
    for (auto const &configItem : configuration) {
        result << "_configItem=" << configItem.first << "_" << configItem.second;
    }
    return result.str();
}

void SplitConcatConcatTest::SetUp() {
    InferenceEngine::Precision netPrecision;
    std::tie(netPrecision, targetDevice, configuration) = this->GetParam();
    auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);

    ov::ParameterVector params {std::make_shared<ov::op::v0::Parameter>(ngPrc, ov::Shape{1, 256})};
    auto relu_start = std::make_shared<ngraph::opset1::Relu>(params[0]);
    auto split_axis_op = std::make_shared<ov::op::v0::Constant>(ov::element::Type_t::i64, ov::Shape{}, std::vector<int64_t>{1});
    auto split = std::make_shared<ov::op::v1::Split>(relu_start, split_axis_op, 2);

    auto const_concat = ngraph::builder::makeConstant(ngPrc, {1, 96}, std::vector<float>{0});
    auto const_concat_2 = ngraph::builder::makeConstant(ngPrc, {1, 96}, std::vector<float>{0});
    auto concat = std::make_shared<ngraph::opset1::Concat>(ngraph::OutputVector{split->output(0), const_concat}, 1);
    auto concat_2 = std::make_shared<ngraph::opset1::Concat>(ngraph::OutputVector{concat, const_concat_2},
                                                             1);
    auto relu = std::make_shared<ngraph::opset1::Relu>(concat_2);
    ngraph::ResultVector resultVector{
            std::make_shared<ngraph::opset1::Result>(relu)
    };
    function = std::make_shared<ngraph::Function>(resultVector, params, "Multiple_connection_split_concat");
}
} // namespace SubgraphTestsDefinitions
