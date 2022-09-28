// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/subgraph/permute_concat_concat_permute.hpp"

#include <debug.h>

#include <ctime>
#include <iterator>

namespace SubgraphTestsDefinitions {
std::string PermuteConcatConcatPermute::getTestCaseName(
    const testing::TestParamInfo<PermuteConcatConcatPermuteTuple>& obj) {
    std::vector<size_t> input_shape;
    InferenceEngine::Precision net_precision;
    std::string targetName;
    std::tie(input_shape, net_precision, targetName) = obj.param;
    std::ostringstream results;

    results << "IS=" << CommonTestUtils::vec2str(input_shape) << "_";
    results << "netPRC=" << net_precision.name() << "_";
    results << "targetDevice=" << targetName << "_";
    return results.str();
}

void PermuteConcatConcatPermute::SetUp() {
    std::srand(std::time(nullptr));

    std::vector<size_t> input_shape;
    InferenceEngine::Precision net_precision;
    std::tie(input_shape, net_precision, targetDevice) = this->GetParam();

    auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(net_precision);

    auto input_param = std::make_shared<ngraph::opset9::Parameter>(ngPrc, ngraph::Shape{input_shape});
    std::vector<size_t> permute_param = {1, 0};
    auto permute_params =
        ngraph::opset9::Constant::create(ngraph::element::i64, ngraph::Shape{permute_param.size()}, permute_param);
    auto permute_1 = std::make_shared<ngraph::opset9::Transpose>(input_param, permute_params);

    auto const_input_1 = CreateConst(input_shape, ngPrc, false);
    auto concat_1 = std::make_shared<ngraph::opset9::Concat>(ngraph::OutputVector{const_input_1, permute_1}, 0);

    auto const_input_2 = CreateConst(input_shape, ngPrc, true);
    auto concat_2 = std::make_shared<ngraph::opset9::Concat>(ngraph::OutputVector{concat_1, const_input_2}, 0);

    auto permute_2 = std::make_shared<ngraph::opset9::Transpose>(concat_2, permute_params);

    function = std::make_shared<ngraph::Function>(permute_2,
                                                  ngraph::ParameterVector{input_param},
                                                  "permute_concat_concat_permute_zero_validation");
    range_ = InferenceEngine::details::product(input_shape);
}

std::shared_ptr<ngraph::opset9::Constant> PermuteConcatConcatPermute::CreateConst(
    const std::vector<size_t>& input_shape,
    const ::ngraph::element::Type& precision,
    bool use_1_as_first_dimension) {
    auto const_input_shape_vec = std::vector<size_t>{};
    if (input_shape.size() == 1) {
        const_input_shape_vec.push_back(input_shape.front());
    } else {
        if (use_1_as_first_dimension) {
            const_input_shape_vec.push_back(1);
            const_input_shape_vec.push_back(input_shape[0]);
        } else {
            const_input_shape_vec.push_back(input_shape[1]);
            const_input_shape_vec.push_back(input_shape[0]);
        }

        const_input_shape_vec.insert(const_input_shape_vec.end(), std::next(input_shape.begin(), 2), input_shape.end());
    }

    const auto const_input_shape = ngraph::Shape{const_input_shape_vec};
    auto const_input_values_size = InferenceEngine::details::product(const_input_shape_vec);
    auto const_input_values = std::vector<size_t>(const_input_values_size, 0);
    return ngraph::opset9::Constant::create(precision, const_input_shape, const_input_values);
}

void PermuteConcatConcatPermute::Validate() {
    if (functionRefs == nullptr) {
        functionRefs = ngraph::clone_function(*function);
    }
    const auto& actual_outputs = GetOutputs();
    IE_ASSERT(actual_outputs.size() == 1);

    auto expected_outputs = CalculateRefs();
    IE_ASSERT(expected_outputs.size() == actual_outputs.size());

    const auto& expected = expected_outputs[0];
    const auto& actual = actual_outputs[0];

    IE_ASSERT(actual->byteSize() == expected.second.size());

    auto memory = InferenceEngine::as<InferenceEngine::MemoryBlob>(actual);
    IE_ASSERT(memory);

    const auto locked_memory = memory->wmap();
    auto precision = actual->getTensorDesc().getPrecision();

    switch (precision) {
    case InferenceEngine::Precision::FP16: {
        IE_ASSERT(expected.first == ngraph::element::f16);
        const auto actual_output_buffer = locked_memory.as<const ngraph::float16*>();
        const auto expected_output_buffer = reinterpret_cast<const ngraph::float16*>(expected.second.data());
        CompareBuffers(expected_output_buffer, actual_output_buffer, actual->size(), threshold);
        break;
    }
    case InferenceEngine::Precision::FP32: {
        IE_ASSERT(expected.first == ngraph::element::f32);
        const auto actual_output_buffer = locked_memory.as<const float*>();
        const auto expected_output_buffer = reinterpret_cast<const float*>(expected.second.data());
        CompareBuffers(expected_output_buffer, actual_output_buffer, actual->size(), threshold);
        break;
    }
    default:
        FAIL() << "Comparator for " << precision << " precision isn't supported";
    }
}

InferenceEngine::Blob::Ptr PermuteConcatConcatPermute::GenerateInput(
    const InferenceEngine::InputInfo& inputInfo) const {
    return FuncTestUtils::createAndFillBlobConsistently(inputInfo.getTensorDesc(), range_, start_, step_);
}

}  // namespace SubgraphTestsDefinitions
