// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/subgraph/split_trivial_permute_concat.hpp"

namespace SubgraphTestsDefinitions {
    std::string SplitTrivialPermuteConcatTest::getTestCaseName(const testing::TestParamInfo<SplitTrivialPermuteConcatTuple>& obj) {
        InferenceEngine::Precision netPrecision;
        std::string targetName;
        std::vector<size_t> inputShape;
        size_t splitAxis;
        size_t concatAxis;
        std::tie(netPrecision, targetName, inputShape, splitAxis, concatAxis, std::ignore) = obj.param;
        std::ostringstream results;

        results << "netPRC=" << netPrecision.name() << "_";
        results << "IS=";
        for (size_t size : inputShape)
            results << size << "_";
        results << "SA=" << splitAxis << "_";
        results << "CA=" << concatAxis << "_";
        results << "targetDevice=" << targetName;
        return results.str();
    }

    void SplitTrivialPermuteConcatTest::SetUp() {
        InferenceEngine::Precision netPrecision;
        std::vector<size_t> inputShape;
        size_t splitAxis;
        size_t concatAxis;
        std::map<std::string, std::string> config;
        std::tie(netPrecision, targetDevice, inputShape, splitAxis, concatAxis, config) = this->GetParam();
        configuration.insert(config.begin(), config.end());
        auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);
        ov::ParameterVector input {std::make_shared<ov::op::v0::Parameter>(ngPrc, ov::Shape(inputShape))};
        auto split_axis_op =
            std::make_shared<ov::op::v0::Constant>(ov::element::Type_t::i64, ov::Shape{}, std::vector<int64_t>{static_cast<int64_t>(splitAxis)});
        auto split = std::make_shared<ov::op::v1::Split>(input[0], split_axis_op, 2);

        auto permute_in_params = std::make_shared<ngraph::opset1::Constant>(ngraph::element::i64,
                                                                            ngraph::Shape{ 4 },
                                                                            ngraph::Shape{ {0, 3, 2, 1} });
        auto permute_0 = std::make_shared<ngraph::opset1::Transpose>(split->output(0), permute_in_params);
        auto permute_1 = std::make_shared<ngraph::opset1::Transpose>(split->output(1), permute_in_params);

        auto concat = std::make_shared<ngraph::opset1::Concat>(ngraph::OutputVector{ permute_0, permute_1 }, concatAxis);
        auto act = ngraph::builder::makeActivation(concat, ngPrc, ngraph::helpers::ActivationTypes::Relu);
        function = std::make_shared<ngraph::Function>(act, input, "split_trivial_permute_concat");
    }
} // namespace SubgraphTestsDefinitions
