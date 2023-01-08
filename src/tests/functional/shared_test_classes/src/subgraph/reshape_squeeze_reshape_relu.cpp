// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <debug.h>
#include "shared_test_classes/subgraph/reshape_squeeze_reshape_relu.hpp"

namespace SubgraphTestsDefinitions {
    std::string ReshapeSqueezeReshapeRelu::getTestCaseName(const testing::TestParamInfo<ReshapeSqueezeReshapeReluTuple> &obj) {
        ShapeAxesTuple squeezeShape;
        InferenceEngine::Precision netPrecision;
        std::string targetName;
        ngraph::helpers::SqueezeOpType opType;
        std::tie(squeezeShape, netPrecision, targetName, opType) = obj.param;
        std::ostringstream results;
        results << "OpType=" << opType;
        results << "IS=" << CommonTestUtils::vec2str(squeezeShape.first) << "_";
        results << "indices=" << CommonTestUtils::vec2str(squeezeShape.second) << "_";
        results << "netPRC=" << netPrecision.name() << "_";
        results << "targetDevice=" << targetName << "_";
        return results.str();
    }

    void ReshapeSqueezeReshapeRelu::SetUp() {
        ShapeAxesTuple squeezeShape;
        InferenceEngine::Precision netPrecision;
        ngraph::helpers::SqueezeOpType opType;
        std::tie(squeezeShape, netPrecision, targetDevice, opType) = this->GetParam();
        const std::size_t input_dim = InferenceEngine::details::product(squeezeShape.first);
        auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);
        std::vector<size_t> shape_input{1, input_dim};
        auto input = ngraph::builder::makeParams(ngPrc, {shape_input});
        auto reshape1_pattern = std::make_shared<ngraph::op::Constant>(ngraph::element::i64,
                                                                       ngraph::Shape{squeezeShape.first.size()},
                                                                       squeezeShape.first);
        auto reshape1 = std::make_shared<ngraph::op::v1::Reshape>(input[0], reshape1_pattern, false);
        auto squeeze = ngraph::builder::makeSqueezeUnsqueeze(reshape1, ngraph::element::i64, squeezeShape.second, opType);
        auto reshape2_pattern = std::make_shared<ngraph::op::Constant>(ngraph::element::i64,
                                                                       ngraph::Shape{2},
                                                                       std::vector<size_t>{1, input_dim});
        auto reshape2 = std::make_shared<ngraph::op::v1::Reshape>(squeeze, reshape2_pattern, false);
        auto func = std::make_shared<ngraph::opset1::Relu>(reshape2);
        std::string squeezeType;

        function = std::make_shared<ngraph::Function>(func, input, "reshape_squeeze_reshape_relu");
    }
} // namespace SubgraphTestsDefinitions
