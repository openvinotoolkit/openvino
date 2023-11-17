// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/subgraph/split_concat_multi_inputs.hpp"
#include "ov_models/builders.hpp"

namespace SubgraphTestsDefinitions {

std::string SplitConcatMultiInputsTest::getTestCaseName(testing::TestParamInfo<SplitConcatMultiInputsParams> obj) {
    std::vector<size_t> inputShape;
    size_t splitsNum;
    std::map<std::string, std::string> config;
    InferenceEngine::Precision netPrecision;
    std::string targetName;
    bool withFC;
    std::tie(netPrecision, targetName, config, inputShape, splitsNum, withFC) = obj.param;
    std::ostringstream result;
    result << "IS=" << ov::test::utils::vec2str(inputShape) << "_";
    result << "SplitsN=" << splitsNum << "_";
    result << "netPRC=" << netPrecision.name() << "_";
    result << "trgDev=" << targetName << "_";
    result << "FC=" << withFC;
    for (auto const& configItem : config) {
        result << "_configItem=" << configItem.first << "_" << configItem.second;
    }
    return result.str();
}

void SplitConcatMultiInputsTest::SetUp() {
    std::vector<size_t> inputShape;
    size_t splitsNum;
    std::map<std::string, std::string> tempConfig;
    InferenceEngine::Precision netPrecision;
    bool withFC;
    std::tie(netPrecision, targetDevice, tempConfig, inputShape, splitsNum, withFC) = this->GetParam();
    configuration.insert(tempConfig.begin(), tempConfig.end());

    inputShape[1] *= splitsNum;
    auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);
    ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(ngPrc, ov::Shape(inputShape))};

    auto split_axis_op = std::make_shared<ov::op::v0::Constant>(ov::element::Type_t::i64, ov::Shape{}, std::vector<int64_t>{1});
    auto split = std::make_shared<ov::op::v1::Split>(params[0], split_axis_op, splitsNum);

    ngraph::OutputVector concatInputs = split->outputs();

    auto concat = std::make_shared<ngraph::opset7::Concat>(concatInputs, 1);

    if (withFC) {
        auto mul_const = ngraph::builder::makeConstant<float>(ngPrc, { 10, inputShape[1] },
            ov::test::utils::generate_float_numbers(10 * inputShape[1], -0.2f, 0.2f), false);
        auto matmul = std::make_shared<ngraph::op::MatMul>(concat, mul_const, false, true);
        function = std::make_shared<ngraph::Function>(matmul, params, "SplitConcatMultiInputs");
    } else {
        function = std::make_shared<ngraph::Function>(concat, params, "SplitConcatMultiInputs");
    }
}

InferenceEngine::Blob::Ptr SplitConcatMultiInputsTest::GenerateInput(const InferenceEngine::InputInfo &info) const {
    return FuncTestUtils::createAndFillBlob(info.getTensorDesc(), inputDataMax - inputDataMin, inputDataMin, 1 / inputDataResolution,
                                            seed);
}
}  // namespace SubgraphTestsDefinitions
