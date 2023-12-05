// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <transformations/op_conversions/lstm_cell_decomposition.hpp>

#include "ov_models/builders.hpp"
#include "shared_test_classes/subgraph/memory_eltwise_reshape_concat.hpp"

namespace SubgraphTestsDefinitions {

std::string MemoryEltwiseReshapeConcatTest::getTestCaseName(const testing::TestParamInfo<memoryEltwiseReshapeConcatParams> &obj) {
    std::string targetDevice;
    InferenceEngine::Precision netPrecision;
    size_t inputSize;
    size_t concatSize;
    std::map<std::string, std::string> config;
    std::tie(targetDevice, netPrecision, inputSize, concatSize, config) = obj.param;
    std::ostringstream result;

    result << "netPrecision=" << netPrecision.name() << "_";
    result << "IS=" << inputSize << "_";
    result << "CS=" << concatSize << "_";
    result << "targetDevice=" << targetDevice;
    return result.str();
}

void MemoryEltwiseReshapeConcatTest::SetUp() {
    InferenceEngine::Precision netPrecision;
    std::map<std::string, std::string> config;
    std::tie(targetDevice, netPrecision, inputSize, concatSize, config) = this->GetParam();
    configuration.insert(config.begin(), config.end());
    ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);

    const int seed = 0;
    std::mt19937 gen(seed);

    auto generateFloatNumbers = [gen](std::size_t vec_len, float min, float max) mutable {
        std::vector<float> res;

        std::uniform_real_distribution<float> dist(min, max);
        for (std::size_t i = 0; i < vec_len; i++)
            res.emplace_back(static_cast<float>(dist(gen)));

        return res;
    };

    memory_init = generateFloatNumbers(inputSize * concatSize, -1.0f, 1.0f);
    concat_vals = generateFloatNumbers(concatSize, 12.0f, 14.0f);
}

void MemoryEltwiseReshapeConcatTest::initTestModel() {
    InferenceEngine::SizeVector input_dims = {1, inputSize * concatSize};
    ov::ParameterVector input_parameter {std::make_shared<ov::op::v0::Parameter>(ngPrc, ov::Shape(input_dims))};

    auto memory_constant = ngraph::builder::makeConstant<float>(ngPrc, input_dims, memory_init);
    memory_constant->set_friendly_name("memory_constant");
    auto memory_read = std::make_shared<ngraph::opset5::ReadValue>(memory_constant, "memory");
    memory_read->set_friendly_name("memory_read");

    auto mul = ngraph::builder::makeEltwise(input_parameter[0], memory_read, ngraph::helpers::EltwiseTypes::MULTIPLY);
    mul->set_friendly_name("multiplication");

    auto memory_write = std::make_shared<ngraph::opset5::Assign>(mul, "memory");
    memory_write->set_friendly_name("memory_write");

    auto reshape_1_pattern = std::make_shared<ngraph::opset5::Constant>(ngraph::element::i64, ngraph::Shape{2}, std::vector<size_t>({inputSize, concatSize}));
    reshape_1_pattern->set_friendly_name("reshape_pattern");
    auto reshape_1 = std::make_shared<ngraph::opset5::Reshape>(mul, reshape_1_pattern, false);
    reshape_1->set_friendly_name("reshape");

    auto concat_constant = ngraph::builder::makeConstant(ngPrc, {1, concatSize}, concat_vals);
    concat_constant->set_friendly_name("concat_constant");

    auto concat = std::make_shared<ov::op::v0::Concat>(ov::NodeVector{concat_constant, reshape_1}, 0);

    memory_write->add_control_dependency(memory_read);
    concat->add_control_dependency(memory_write);

    auto final_reshape_pattern = std::make_shared<ngraph::opset5::Constant>(ngraph::element::i64, ngraph::Shape{4},
                                                                        std::vector<size_t>({1, 1, inputSize + 1, concatSize}));
    auto final_reshape = std::make_shared<ngraph::opset5::Reshape>(concat, final_reshape_pattern, false);

    function = std::make_shared<ngraph::Function>(final_reshape, input_parameter, "memory_multiply_reshape_concat");
}

void MemoryEltwiseReshapeConcatTest::initNgraphFriendlyModel() {
    InferenceEngine::SizeVector input_dims = {1, inputSize * concatSize};
    ov::ParameterVector input_parameter {std::make_shared<ov::op::v0::Parameter>(ngPrc, ov::Shape(input_dims))};

    auto memory_constant = ngraph::builder::makeConstant<float>(ngPrc, input_dims, memory_init);
    memory_constant->set_friendly_name("memory_constant");

    auto mul = ngraph::builder::makeEltwise(input_parameter[0], memory_constant, ngraph::helpers::EltwiseTypes::MULTIPLY);
    mul->set_friendly_name("multiplication");

    auto reshape_pattern = std::make_shared<ngraph::opset5::Constant>(ngraph::element::i64, ngraph::Shape{3}, std::vector<size_t>({1, inputSize, concatSize}));
    reshape_pattern->set_friendly_name("reshape_pattern");
    auto reshape = std::make_shared<ngraph::opset5::Reshape>(mul, reshape_pattern, false);
    reshape->set_friendly_name("reshape");

    auto squeeze_const = std::make_shared<ngraph::opset5::Constant>(ngraph::element::i64, ngraph::Shape{1}, 0);
    squeeze_const->set_friendly_name("squeeze_const");
    auto squeeze = std::make_shared<ngraph::opset5::Squeeze>(reshape, squeeze_const);
    squeeze->set_friendly_name("squeeze");

    auto concat_constant = ngraph::builder::makeConstant(ngPrc, {1, concatSize}, concat_vals);
    concat_constant->set_friendly_name("concat_constant");

    auto concat = std::make_shared<ov::op::v0::Concat>(ov::NodeVector{concat_constant, squeeze}, 0);

    function = std::make_shared<ngraph::Function>(concat, input_parameter, "memory_multiply_reshape_concat");
}

void MemoryEltwiseReshapeConcatTest::LoadNetwork() {
    LayerTestsUtils::LayerTestsCommon::LoadNetwork();
    inferRequest = executableNetwork.CreateInferRequest();
}

void MemoryEltwiseReshapeConcatTest::Infer() {
    ConfigureInferRequest();
    inferRequest.Infer();
}

void MemoryEltwiseReshapeConcatTest::Run() {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    initTestModel();
    LoadNetwork();

    InferenceEngine::TensorDesc state_description(InferenceEngine::Precision::FP32,
                                                  InferenceEngine::SizeVector({1, inputSize * concatSize}),
                                                  InferenceEngine::Layout::NC);

    IE_SUPPRESS_DEPRECATED_START
    auto states = inferRequest.QueryState();
    auto state_values_blob = FuncTestUtils::createAndFillBlobWithFloatArray(state_description,
                                                                            memory_init.data(), memory_init.size());
    states[0].SetState(state_values_blob);
    IE_SUPPRESS_DEPRECATED_END
    GenerateInputs();
    Infer();
    initNgraphFriendlyModel();
    Validate();
}
}  // namespace SubgraphTestsDefinitions
