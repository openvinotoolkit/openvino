// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/subgraph/copy_before_squeeze.hpp"

namespace SubgraphTestsDefinitions {
    std::string CopyBeforeSqueezeTest::getTestCaseName(const testing::TestParamInfo<CopyBeforeSqueezeTuple>& obj) {
        InferenceEngine::Precision netPrecision;
        std::string targetName;
        std::vector<size_t> inputShape;
        std::tie(netPrecision, targetName, inputShape, std::ignore) = obj.param;
        std::ostringstream results;

        results << "netPRC=" << netPrecision.name() << "_";
        results << "IS=" << ov::test::utils::vec2str(inputShape) << "_";
        results << "targetDevice=" << targetName;
        return results.str();
    }

    void CopyBeforeSqueezeTest::SetUp() {
        InferenceEngine::Precision netPrecision;
         std::vector<size_t> inputShape;
        std::map<std::string, std::string> config;
        std::tie(netPrecision, targetDevice, inputShape, config) = this->GetParam();
        configuration.insert(config.begin(), config.end());
        auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);

        ov::ParameterVector input {std::make_shared<ov::op::v0::Parameter>(ngPrc, ov::Shape(inputShape))};
        auto reshape_0_pattern = std::make_shared<ngraph::op::Constant>(ngraph::element::i64,
                                                                        ngraph::Shape{3},
                                                                        std::vector<size_t>{1, inputShape[1] / 64, 64});
        auto reshape_0 = std::make_shared<ngraph::op::v1::Reshape>(input[0], reshape_0_pattern, false);
        auto relu = std::make_shared<ngraph::opset1::Relu>(reshape_0);

        auto constant_squeeze = std::make_shared<ngraph::op::Constant>(ngraph::element::Type_t::i64, ngraph::Shape{1}, std::vector<size_t>{0});
        auto reshape_pattern = std::make_shared<ngraph::op::Constant>(ngraph::element::i64,
                                                                      ngraph::Shape{2},
                                                                      std::vector<size_t>{1, inputShape[1]});
        auto squeeze_1 = std::make_shared<ngraph::op::Squeeze>(relu, constant_squeeze);
        auto reshape_1 = std::make_shared<ngraph::op::v1::Reshape>(squeeze_1, reshape_pattern, false);
        auto squeeze_2 = std::make_shared<ngraph::op::Squeeze>(relu, constant_squeeze);
        auto reshape_2 = std::make_shared<ngraph::op::v1::Reshape>(squeeze_2, reshape_pattern, false);

        auto concat = std::make_shared<ngraph::opset1::Concat>(ngraph::OutputVector{reshape_1, reshape_2}, 1);
        function = std::make_shared<ngraph::Function>(concat, input, "copy_before_squeeze");
    }
} // namespace SubgraphTestsDefinitions
