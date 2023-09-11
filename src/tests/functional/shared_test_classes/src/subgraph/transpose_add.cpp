// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/subgraph/transpose_add.hpp"

namespace SubgraphTestsDefinitions {
std::string TransposeAdd::getTestCaseName(testing::TestParamInfo<TransposeAddParams> obj) {
    InferenceEngine::Precision netPrecision;
    std::string targetName;
    std::vector<size_t> input_shape;
    std::map<std::string, std::string> configuration;

    std::tie(netPrecision, targetName, input_shape, configuration) = obj.param;
    std::ostringstream results;

    results << "IS=" << ov::test::utils::vec2str(std::vector<size_t>(input_shape.begin(), input_shape.end())) << "_";
    results << "netPRC=" << netPrecision.name() << "_";
    results << "targetDevice=" << targetName;
    return results.str();
}

void TransposeAdd::SetUp() {
    InferenceEngine::Precision netPrecision;
    std::vector<size_t> input_shape;
    std::map<std::string, std::string> additional_config;

    std::tie(netPrecision, targetDevice, input_shape, additional_config) = this->GetParam();
    GTEST_ASSERT_GE(input_shape.size(), 2);

    configuration.insert(additional_config.begin(), additional_config.end());

    auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);
    ov::ParameterVector params {std::make_shared<ov::op::v0::Parameter>(ngPrc, ov::Shape(input_shape))};

    ngraph::Shape permute_order(input_shape.size());
    std::iota(std::begin(permute_order), std::end(permute_order), 0);
    std::iter_swap(std::end(permute_order) - 2, std::end(permute_order) - 1);
    auto transpose_in_params = std::make_shared<ngraph::opset8::Constant>(ngraph::element::i64,
        ngraph::Shape{permute_order.size()}, permute_order);
    auto transpose_in = std::make_shared<ngraph::opset8::Transpose>(params[0], transpose_in_params);

    auto add_const = ngraph::builder::makeConstant<float>(ngPrc, transpose_in->get_output_shape(0), {}, true);
    auto add = std::make_shared<ngraph::opset8::Add>(transpose_in, add_const);

    function = std::make_shared<ngraph::Function>(add, params, "transpose_add");
}

} // namespace SubgraphTestsDefinitions
