// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/subgraph/concat_multi_input.hpp"

namespace SubgraphTestsDefinitions {

std::string ConcatMultiInput::getTestCaseName(const testing::TestParamInfo<concatMultiParams>& obj) {
    std::vector<std::vector<size_t>> inputShapes;
    InferenceEngine::Precision netPrecision;
    std::string targetDevice;
    std::map<std::string, std::string> additional_config;
    std::tie(inputShapes, netPrecision, targetDevice, additional_config) = obj.param;

    std::ostringstream result;
    result << "IS=" << ov::test::utils::vec2str(inputShapes) << "_";
    result << "netPRC=" << netPrecision.name() << "_";
    result << "targetDevice=" << targetDevice;
    for (auto const& configItem : additional_config) {
        result << "_configItem=" << configItem.first << "_" << configItem.second;
    }

    return result.str();
}

void ConcatMultiInput::SetUp() {
    InferenceEngine::Precision netPrecision;
    std::map<std::string, std::string> additional_config;
    std::tie(inputShapes, netPrecision, targetDevice, additional_config) = this->GetParam();
    configuration.insert(additional_config.begin(), additional_config.end());

    ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);
    paramSize = { 1, 0 };
    for (const auto& val : inputShapes) {
        paramSize[1] += val[1];
    }
}

void ConcatMultiInput::GenerateStridedSliceModel() {
    ov::ParameterVector params {std::make_shared<ov::op::v0::Parameter>(ngPrc, ov::Shape(paramSize))};
    auto stride = std::make_shared<ngraph::op::Constant>(ngraph::element::i64, ngraph::Shape{ 2 }, std::vector<int64_t>{ 1, 1 });

    std::vector<int64_t> newAxis = { 0, 0 };
    std::vector<int64_t> begin_mask = { 0, 0 };
    std::vector<int64_t> end_mask = { 0, 0 };
    std::vector<std::shared_ptr<ngraph::opset1::StridedSlice>> ssArray;
    ngraph::OutputVector concatInput;

    auto relu = std::make_shared<ngraph::opset1::Relu>(params[0]);
    std::vector<int64_t> startOffset = { 0, 0 };
    for (size_t i = 0; i < inputShapes.size(); ++i) {
        std::vector<int64_t> shape = { static_cast<int64_t>(inputShapes[i][0]),
                                       static_cast<int64_t>(inputShapes[i][1]) };
        std::vector<int64_t> endoffset = { static_cast<int64_t>(inputShapes[i][0]) + startOffset[0],
                                           static_cast<int64_t>(inputShapes[i][1]) + startOffset[1]};
        auto begin = std::make_shared<ngraph::op::Constant>(ngraph::element::i64, ngraph::Shape{ 2 }, startOffset);
        auto end = std::make_shared<ngraph::op::Constant>(ngraph::element::i64, ngraph::Shape{ 2 }, endoffset);
        auto ss = std::make_shared<ngraph::opset1::StridedSlice>(relu, begin, end, stride, begin_mask, end_mask, newAxis);
        ssArray.push_back(ss);
        concatInput.push_back(ssArray[i]);

        startOffset[1] += shape[1];
    }

    auto concat = std::make_shared<ngraph::opset1::Concat>(concatInput, 1);

    ngraph::ResultVector results{ std::make_shared<ngraph::opset1::Result>(concat) };
    function = std::make_shared<ngraph::Function>(results, params, "ConcatMultiInput");
}

void ConcatMultiInput::GenerateConstOnlyModel() {
    ngraph::OutputVector concatInputs;

    const int seed = 0;
    std::mt19937 gen(seed);

    auto generateFloatNumbers = [gen](std::size_t vec_len, float min, float max) mutable {
        std::vector<float> res;

        std::uniform_real_distribution<float> dist(min, max);
        for (std::size_t i = 0; i < vec_len; i++)
            res.emplace_back(static_cast<float>(dist(gen)));

        return res;
    };
    ov::ParameterVector input_vector;
    for (size_t i = 0; i < inputShapes.size(); ++i) {
        size_t total_size = 1;
        for (auto dim : inputShapes[i]) {
            total_size *= dim;
        }
        if (i == 0) {
            input_vector = ov::ParameterVector{std::make_shared<ov::op::v0::Parameter>(ngPrc, ov::Shape{1, total_size})};
            auto relu = ngraph::builder::makeActivation(input_vector[0], ngPrc, ngraph::helpers::ActivationTypes::Relu);
            concatInputs.push_back(relu);
        } else {
            auto min_max = (i % 2 == 0) ? 2 : 30;
            auto const_values = generateFloatNumbers(total_size, -min_max, min_max);
            auto const_node = ngraph::builder::makeConstant(ngPrc, {1, total_size}, const_values);
            concatInputs.push_back(const_node);
        }
    }

    auto concat = std::make_shared<ov::op::v0::Concat>(concatInputs, 1);

    ngraph::ResultVector results{ std::make_shared<ngraph::opset1::Result>(concat) };
    function = std::make_shared<ngraph::Function>(results, input_vector, "ConcatConstOnly");
}

void ConcatMultiInput::GenerateMemoryModel() {
    int axis = 1;
    ov::ParameterVector input{std::make_shared<ov::op::v0::Parameter>(ngPrc, ov::Shape(inputShapes[0]))};

    auto variable = std::make_shared<ngraph::Variable>(ngraph::VariableInfo{ngraph::PartialShape::dynamic(),
                                                                            ngraph::element::dynamic, "concat_input_memory"});
    auto mem_i = std::make_shared<ngraph::opset8::Constant>(ngPrc, inputShapes[0]);
    auto mem_r = std::make_shared<ngraph::opset8::ReadValue>(mem_i, variable);

    ngraph::OutputVector concat_input;
    concat_input.push_back(mem_r);
    concat_input.push_back(input.at(0));
    auto concat = std::make_shared<ngraph::opset8::Concat>(concat_input, axis);

    auto mem_w = std::make_shared<ngraph::opset8::Assign>(input.at(0), variable);

    auto res = std::make_shared<ngraph::opset8::Result>(concat);
    function = std::make_shared<ngraph::Function>(ngraph::ResultVector{res}, ngraph::SinkVector{mem_w}, input, "ConcatMemory");
}

}  // namespace SubgraphTestsDefinitions
