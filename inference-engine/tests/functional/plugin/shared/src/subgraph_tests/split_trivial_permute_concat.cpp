// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include <tuple>
#include <string>
#include <vector>
#include <memory>
#include <debug.h>
#include "common_test_utils/common_utils.hpp"
#include "functional_test_utils/precision_utils.hpp"
#include "functional_test_utils/skip_tests_config.hpp"
#include "subgraph_tests/split_trivial_permute_concat.hpp"
#include "ngraph_functions/utils/ngraph_helpers.hpp"

namespace LayerTestsDefinitions {
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
        auto input = ngraph::builder::makeParams(ngPrc, { inputShape });
        auto split = ngraph::builder::makeSplit(input[0], ngPrc, 2, splitAxis);

        auto permute_in_params = std::make_shared<ngraph::opset1::Constant>(ngraph::element::i64,
                                                                            ngraph::Shape{ 4 },
                                                                            ngraph::Shape{ {0, 3, 2, 1} });
        auto permute_0 = std::make_shared<ngraph::opset1::Transpose>(split->output(0), permute_in_params);
        auto permute_1 = std::make_shared<ngraph::opset1::Transpose>(split->output(1), permute_in_params);

        auto concat = std::make_shared<ngraph::opset1::Concat>(ngraph::OutputVector{ split->output(0), split->output(1) }, concatAxis);
        auto act = ngraph::builder::makeActivation(concat, ngPrc, ngraph::helpers::ActivationTypes::Relu);
        function = std::make_shared<ngraph::Function>(act, input, "split_trivial_permute_concat");
    }

    TEST_P(SplitTrivialPermuteConcatTest, CompareWithRefs) {
        Run();
    };
} // namespace LayerTestsDefinitions
