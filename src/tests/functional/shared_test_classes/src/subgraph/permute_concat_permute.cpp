// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/subgraph/permute_concat_permute.hpp"

#include <debug.h>

#include <cstdlib>
#include <ctime>
#include <iterator>

#include "ov_models/builders.hpp"

namespace SubgraphTestsDefinitions {
std::string PermuteConcatPermute::getTestCaseName(const testing::TestParamInfo<PermuteConcatPermuteTuple>& obj) {
    std::vector<std::vector<size_t>> input;
    InferenceEngine::Precision netPrecision;
    std::string targetName;
    std::tie(input, netPrecision, targetName) = obj.param;
    std::ostringstream results;

    results << "IS=" << ov::test::utils::vec2str(input[0]) << "_";
    results << "netPRC=" << netPrecision.name() << "_";
    results << "targetDevice=" << targetName << "_";
    return results.str();
}

void PermuteConcatPermute::SetUp() {
    std::srand(std::time(nullptr));

    std::vector<std::vector<size_t>> inputs;
    InferenceEngine::Precision netPrecision;
    std::tie(inputs, netPrecision, targetDevice) = this->GetParam();
    auto input_shape = inputs[0];
    auto permute_1_param = inputs[1];
    auto permute_2_param = inputs[2];

    auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);

    auto input_param = std::make_shared<ngraph::opset9::Parameter>(ngPrc, ngraph::Shape{input_shape});
    auto permute_params_1 =
        ngraph::opset9::Constant::create(ngraph::element::i64, ngraph::Shape{permute_1_param.size()}, permute_1_param);

    auto permute_1 = std::make_shared<ngraph::opset9::Transpose>(input_param, permute_params_1);

    auto const_input_shape_vec = std::vector<size_t>{1};
    const_input_shape_vec.insert(const_input_shape_vec.end(), input_shape.begin(), std::prev(input_shape.end()));
    const auto constinput_shape = ngraph::Shape{const_input_shape_vec};
    auto const_input_values_size = InferenceEngine::details::product(const_input_shape_vec);
    auto const_input_values = std::vector<size_t>(const_input_values_size, 0);

    auto const_input_1 = ngraph::opset9::Constant::create(ngPrc, constinput_shape, const_input_values);
    auto const_input_2 = ngraph::opset9::Constant::create(ngPrc, constinput_shape, const_input_values);
    auto const_input_3 = ngraph::opset9::Constant::create(ngPrc, constinput_shape, const_input_values);

    auto concat = std::make_shared<ngraph::opset9::Concat>(
        ngraph::OutputVector{const_input_1, const_input_2, permute_1, const_input_3},
        0);
    auto permute_params_2 =
        ngraph::opset9::Constant::create(ngraph::element::i64, ngraph::Shape{permute_2_param.size()}, permute_2_param);
    auto permute_2 = std::make_shared<ngraph::opset9::Transpose>(concat, permute_params_2);

    function =
        std::make_shared<ngraph::Function>(permute_2, ngraph::ParameterVector{input_param}, "permute_concat_permute");
    range_ = InferenceEngine::details::product(input_shape);
}

InferenceEngine::Blob::Ptr PermuteConcatPermute::GenerateInput(const InferenceEngine::InputInfo& inputInfo) const {
    return FuncTestUtils::createAndFillBlobConsistently(inputInfo.getTensorDesc(), range_, start_, step_);
}
}  // namespace SubgraphTestsDefinitions
