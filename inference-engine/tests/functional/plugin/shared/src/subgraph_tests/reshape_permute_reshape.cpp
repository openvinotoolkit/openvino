// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <tuple>
#include <string>
#include <vector>
#include <memory>
#include <debug.h>
#include "common_test_utils/common_utils.hpp"
#include "functional_test_utils/precision_utils.hpp"
#include "functional_test_utils/skip_tests_config.hpp"
#include "subgraph_tests/reshape_permute_reshape.hpp"

namespace LayerTestsDefinitions {
    std::string ReshapePermuteReshape::getTestCaseName(const testing::TestParamInfo<ReshapePermuteReshapeTuple> &obj) {
        std::vector<std::vector<size_t >> input;
        InferenceEngine::Precision netPrecision;
        std::string targetName;
        std::tie(input, netPrecision, targetName) = obj.param;
        std::ostringstream results;

        results << "IS=" << CommonTestUtils::vec2str(input[0]) << "_";
        results << "netPRC=" << netPrecision.name() << "_";
        results << "targetDevice=" << targetName << "_";
        return results.str();
    }

    void ReshapePermuteReshape::SetUp() {
        std::vector<std::vector<size_t >> inputs;
        InferenceEngine::Precision netPrecision;
        std::tie(inputs, netPrecision, targetDevice) = this->GetParam();
        const std::size_t input_dim = InferenceEngine::details::product(inputs[0]);
        auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);
        std::vector<size_t> shape_input{1, input_dim};
        auto input = ngraph::builder::makeParams(ngPrc, {shape_input});
        auto reshape1_pattern = std::make_shared<ngraph::op::Constant>(ngraph::element::i64,
                                                                       ngraph::Shape{inputs[0].size()},
                                                                       inputs[0]);
        auto reshape1 = std::make_shared<ngraph::op::v1::Reshape>(input[0], reshape1_pattern, false);
        auto permute_params = std::make_shared<ngraph::opset1::Constant>(ngraph::element::i64,
                                                                         ngraph::Shape{inputs[1].size()},
                                                                         inputs[1]);
        auto permute = std::make_shared<ngraph::opset1::Transpose>(reshape1, permute_params);
        auto reshape2_pattern = std::make_shared<ngraph::op::Constant>(ngraph::element::i64,
                                                                       ngraph::Shape{2},
                                                                       std::vector<size_t>{1, input_dim});
        auto reshape2 = std::make_shared<ngraph::op::v1::Reshape>(permute, reshape2_pattern, false);
        function = std::make_shared<ngraph::Function>(reshape2, input, "reshape_permute_reshape");
    }

    TEST_P(ReshapePermuteReshape, CompareWithRefs){
        Run();
        if (targetDevice == std::string{CommonTestUtils::DEVICE_GPU}) {
            PluginCache::get().reset();
        }    };
} // namespace LayerTestsDefinitions
