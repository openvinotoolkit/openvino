// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/subgraph/multioutput_eltwise_squeeze_eltwise.hpp"

namespace SubgraphTestsDefinitions {
    std::string MultioutputEltwiseReshapeEltwise::getTestCaseName(const testing::TestParamInfo<MultioutputEltwiseReshapeEltwiseTuple> &obj) {
        std::vector<std::vector<size_t>> input;
        InferenceEngine::Precision netPrecision;
        std::string targetName;
        std::map<std::string, std::string> additional_config;
        std::tie(input, netPrecision, targetName, additional_config) = obj.param;
        std::ostringstream results;

        results << "IS=" << ov::test::utils::vec2str(input[0]) << "_";
        results << "netPRC=" << netPrecision.name() << "_";
        results << "targetDevice=" << targetName << "_";
        for (auto const& configItem : additional_config) {
            results << "_configItem=" << configItem.first << "_" << configItem.second;
        }
        return results.str();
    }

    void MultioutputEltwiseReshapeEltwise::SetUp() {
        std::vector<std::vector<size_t>> inputs;
        InferenceEngine::Precision netPrecision;
        std::map<std::string, std::string> additional_config;
        std::tie(inputs, netPrecision, targetDevice, additional_config) = this->GetParam();
        configuration.insert(additional_config.begin(), additional_config.end());
        auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);
        ov::ParameterVector input;
        for (auto&& shape : inputs) {
            input.push_back(std::make_shared<ov::op::v0::Parameter>(ngPrc, ov::Shape(shape)));
        }
        auto eltwise_const = ngraph::builder::makeConstant(ngPrc,
                                                    ngraph::Shape{input[0]->get_shape()},
                                                     std::vector<float>{-1.0f});
        auto eltwise = std::make_shared<ngraph::opset1::Multiply>(input[0], eltwise_const);
        auto squeeze = ngraph::builder::makeSqueezeUnsqueeze(eltwise, ngraph::element::i64, {0}, ngraph::helpers::SqueezeOpType::UNSQUEEZE);
        auto unsqueeze = ngraph::builder::makeSqueezeUnsqueeze(squeeze, ngraph::element::i64, {0}, ngraph::helpers::SqueezeOpType::SQUEEZE);
        auto eltwise_const2 = ngraph::builder::makeConstant(ngPrc, ngraph::Shape{1}, std::vector<float>{1.01f});
        auto eltwise_const3 = ngraph::builder::makeConstant(ngPrc, ngraph::Shape{1}, std::vector<float>{1.01f});
        auto eltwise2 = std::make_shared<ngraph::opset1::Multiply>(eltwise, eltwise_const2);
        auto eltwise3 = std::make_shared<ngraph::opset1::Multiply>(unsqueeze, eltwise_const3);
        ngraph::ResultVector results{std::make_shared<ngraph::opset1::Result>(eltwise2),
                                     std::make_shared<ngraph::opset1::Result>(eltwise3)};
        function = std::make_shared<ngraph::Function>(results, input, "eltwise_reshape_eltwise_multioutput");
    }
} // namespace SubgraphTestsDefinitions
