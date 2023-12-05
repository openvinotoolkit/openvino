// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <debug.h>
#include "shared_test_classes/subgraph/reshape_permute_reshape.hpp"

namespace SubgraphTestsDefinitions {
    std::string ReshapePermuteReshape::getTestCaseName(const testing::TestParamInfo<ReshapePermuteReshapeTuple> &obj) {
        std::vector<std::vector<size_t >> input;
        InferenceEngine::Precision netPrecision;
        std::string targetName;
        std::tie(input, netPrecision, targetName) = obj.param;
        std::ostringstream results;

        results << "IS=" << ov::test::utils::vec2str(input[0]) << "_";
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
        ov::ParameterVector input {std::make_shared<ov::op::v0::Parameter>(ngPrc, ov::Shape(shape_input))};
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
} // namespace SubgraphTestsDefinitions
