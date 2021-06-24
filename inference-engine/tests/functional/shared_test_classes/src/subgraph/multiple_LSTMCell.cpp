// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ie_transformations.hpp"
#include "ngraph/opsets/opset7.hpp"
#include "ngraph/op/util/variable_context.hpp"
#include "ngraph/pass/low_latency.hpp"

#include "ngraph_functions/builders.hpp"
#include "ngraph_functions/utils/ngraph_helpers.hpp"

#include "shared_test_classes/subgraph/multiple_LSTMCell.hpp"

using namespace ngraph;
using namespace opset7;

namespace SubgraphTestsDefinitions {
std::string MultipleLSTMCellTest::getTestCaseName(const testing::TestParamInfo<multipleLSTMCellParams> &obj) {
    std::string targetDevice;
    InferenceEngine::Precision netPrecision;
    size_t inputSize;
    size_t hiddenSize;
    std::map<std::string, std::string> config;
    ngraph::helpers::MemoryTransformation transformation;
    std::tie(transformation, targetDevice, netPrecision, inputSize, hiddenSize, config) = obj.param;
    std::ostringstream result;

    result << "transformation=" << transformation << "_";
    result << "netPrecision=" << netPrecision.name() << "_";
    result << "IS=" << inputSize << "_";
    result << "HS=" << hiddenSize << "_";
    result << "targetDevice=" << targetDevice;
    return result.str();
}

void MultipleLSTMCellTest::SetUp() {
    InferenceEngine::Precision netPrecision;
    std::map<std::string, std::string> config;
    size_t inputSize;
    std::tie(transformation, targetDevice, netPrecision, inputSize, hiddenSize, config) = this->GetParam();
    configuration.insert(config.begin(), config.end());
    auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);

    std::vector<size_t> input_dims { 1, inputSize };
    std::vector<size_t> squeeze_axes {0};
    std::vector<size_t> hidden_memory_dims {1, hiddenSize};
    std::vector<size_t> cell_memory_dims {1, hiddenSize};

    input_bias = CommonTestUtils::generate_float_numbers(inputSize, -0.25f, 0.0f);
    input_weights = CommonTestUtils::generate_float_numbers(inputSize, 0.0f, 0.15f);
    hidden_memory_init = CommonTestUtils::generate_float_numbers(hiddenSize, -0.2f, 0.2f);
    cell_memory_init = CommonTestUtils::generate_float_numbers(hiddenSize, -0.2f, 0.2f);
    weights_vals = CommonTestUtils::generate_float_numbers(4 * hiddenSize * inputSize, -0.1f, 0.1f);
    weights_2_vals = CommonTestUtils::generate_float_numbers(4 * hiddenSize * hiddenSize, -0.1f, 0.1f);
    reccurrenceWeights_vals = CommonTestUtils::generate_float_numbers(4 * hiddenSize * hiddenSize, -0.1f, 0.1f);
    bias_vals = CommonTestUtils::generate_float_numbers(4 * hiddenSize, -0.25f, 0.15f);

    auto input_parameter = builder::makeParams(ngPrc, {input_dims});

    auto input_add_const = builder::makeConstant(ngPrc, input_dims, input_bias);
    auto add = builder::makeEltwise(input_parameter[0], input_add_const, helpers::EltwiseTypes::ADD);

    auto input_mul_const = builder::makeConstant(ngPrc, input_dims, input_weights);
    auto mul = builder::makeEltwise(add, input_mul_const, helpers::EltwiseTypes::MULTIPLY);

    auto unsqueeze_input_const = std::make_shared<Constant>(element::i64, Shape{1}, squeeze_axes);
    auto unsqueeze_input = std::make_shared<Unsqueeze>(mul, unsqueeze_input_const);

    auto permute_in_params = std::make_shared<Constant>(element::i64, Shape{3}, Shape{{1, 0, 2}});
    auto permute_in = std::make_shared<Transpose>(unsqueeze_input, permute_in_params);

    auto cell_memory_constant = builder::makeConstant<float>(ngPrc, cell_memory_dims, cell_memory_init);
    auto var_cell =
            std::make_shared<Variable>(VariableInfo{PartialShape::dynamic(), element::dynamic, "cell_state_1"});
    auto var_hidden =
            std::make_shared<Variable>(VariableInfo{PartialShape::dynamic(), element::dynamic, "hidden_state_1"});
    auto cell_memory_read = std::make_shared<ReadValue>(cell_memory_constant, var_cell);
    cell_memory_read->set_friendly_name("cell_memory");

    auto hidden_memory_constant = builder::makeConstant<float>(ngPrc, hidden_memory_dims, hidden_memory_init);
    auto hidden_memory_read = std::make_shared<ReadValue>(hidden_memory_constant, var_hidden);
    hidden_memory_read->set_friendly_name("hidden_memory");

    // Body - inputs
    auto X = std::make_shared<Parameter>(ngPrc, Shape{1, 1, inputSize});
    auto H_t = std::make_shared<Parameter>(ngPrc, Shape{1, hiddenSize});
    auto C_t = std::make_shared<Parameter>(ngPrc, Shape{1, hiddenSize});
    // Body - layers
    auto squeeze_const = std::make_shared<Constant>(element::i64, Shape{1}, squeeze_axes);
    auto squeeze = std::make_shared<Squeeze>(X, squeeze_const);

    auto weightsNode = builder::makeConstant<float>(ngPrc, { 4 * hiddenSize, inputSize }, weights_vals);
    auto reccurrenceWeightsNode = builder::makeConstant<float>(ngPrc, { 4 * hiddenSize, hiddenSize }, reccurrenceWeights_vals);
    auto biasNode = builder::makeConstant<float>(ngPrc, {4 * hiddenSize}, bias_vals);
    auto lstm = std::make_shared<LSTMCell>(squeeze, H_t, C_t, weightsNode, reccurrenceWeightsNode, biasNode, hiddenSize);

    auto unsqueeze_const = std::make_shared<Constant>(element::i64, Shape{1}, squeeze_axes);
    auto unsqueeze = std::make_shared<Unsqueeze>(lstm->output(0), unsqueeze_const);
    // body - outputs
    auto H_o = lstm->output(0);
    auto C_o = lstm->output(1);
    auto unsqueeze_o = unsqueeze->output(0);

    auto body = std::make_shared<Function>(OutputVector{unsqueeze_o, H_o, C_o}, ParameterVector {X, H_t, C_t});
    // TI construction
    auto tensor_iterator = std::make_shared<TensorIterator>();
    tensor_iterator->set_body(body);
    tensor_iterator->set_invariant_input(X, permute_in);
    tensor_iterator->set_merged_input(H_t, hidden_memory_read, H_o);
    tensor_iterator->set_merged_input(C_t, cell_memory_read, C_o);

    auto out_unsqueeze = tensor_iterator->get_iter_value(unsqueeze_o, -1);
    auto out_hidden = tensor_iterator->get_iter_value(H_o, -1);
    auto out_cell = tensor_iterator->get_iter_value(C_o, -1);

    out_hidden.get_tensor().set_element_type(ngPrc);
    out_cell.get_tensor().set_element_type(ngPrc);

    auto cell_memory_write = std::make_shared<Assign>(out_cell, var_cell);
    auto hidden_memory_write = std::make_shared<Assign>(out_hidden, var_hidden);

    auto first_reshape_pattern = std::make_shared<Constant>(element::i64,
                                                                        Shape{4}, std::vector<size_t>({1, 1, 1, hiddenSize}));
    auto first_reshape = std::make_shared<Reshape>(out_unsqueeze, first_reshape_pattern, false);
    // End of TI 1

    auto inbetween_squeeze_const = std::make_shared<Constant>(element::i64, Shape{1}, squeeze_axes);
    auto inbetween_squeeze = std::make_shared<Squeeze>(first_reshape, inbetween_squeeze_const);

    // Second TI
    auto var_cell_2 =
            std::make_shared<Variable>(VariableInfo{PartialShape::dynamic(), element::dynamic, "cell_state_2"});
    auto var_hidden_2 =
            std::make_shared<Variable>(VariableInfo{PartialShape::dynamic(), element::dynamic, "hidden_state_2"});
    auto cell_memory_2_constant = builder::makeConstant<float>(ngPrc, cell_memory_dims, cell_memory_init);
    auto cell_memory_2_read = std::make_shared<ReadValue>(cell_memory_2_constant, var_cell_2);
    cell_memory_2_read->set_friendly_name("cell_memory_2");

    auto hidden_memory_2_constant = builder::makeConstant<float>(ngPrc, hidden_memory_dims, hidden_memory_init);
    auto hidden_memory_2_read = std::make_shared<ReadValue>(hidden_memory_2_constant, var_hidden_2);
    hidden_memory_2_read->set_friendly_name("hidden_memory_2");

    // Body - inputs
    auto X_2 = std::make_shared<Parameter>(ngPrc, Shape{1, 1, hiddenSize});
    auto H_t_2 = std::make_shared<Parameter>(ngPrc, Shape{1, hiddenSize});
    auto C_t_2 = std::make_shared<Parameter>(ngPrc, Shape{1, hiddenSize});
    // Body - layers
    auto squeeze_2_const = std::make_shared<Constant>(element::i64, Shape{1}, squeeze_axes);
    auto squeeze_2 = std::make_shared<Squeeze>(X_2, squeeze_2_const);

    auto weightsNode_2 = builder::makeConstant<float>(ngPrc, { 4 * hiddenSize, hiddenSize }, weights_2_vals);
    auto reccurrenceWeightsNode_2 = builder::makeConstant<float>(ngPrc, { 4 * hiddenSize, hiddenSize }, reccurrenceWeights_vals);
    auto biasNode_2 = builder::makeConstant<float>(ngPrc, {4 * hiddenSize}, bias_vals);
    auto lstm_2 = std::make_shared<LSTMCell>(squeeze_2, H_t_2, C_t_2, weightsNode_2, reccurrenceWeightsNode_2, biasNode_2, hiddenSize);

    auto unsqueeze_2_const = std::make_shared<Constant>(element::i64, Shape{1}, squeeze_axes);
    auto unsqueeze_2 = std::make_shared<Unsqueeze>(lstm_2->output(0), unsqueeze_2_const);
    // body - outputs
    auto H_o_2 = lstm_2->output(0);
    auto C_o_2 = lstm_2->output(1);
    auto unsqueeze_o_2 = unsqueeze_2->output(0);

    auto body_2 = std::make_shared<Function>(OutputVector{unsqueeze_o_2, H_o_2, C_o_2}, ParameterVector {X_2, H_t_2, C_t_2});
    // TI construction
    auto tensor_iterator_2 = std::make_shared<TensorIterator>();
    tensor_iterator_2->set_body(body_2);
    tensor_iterator_2->set_invariant_input(X_2, inbetween_squeeze);
    tensor_iterator_2->set_merged_input(H_t_2, hidden_memory_2_read, H_o_2);
    tensor_iterator_2->set_merged_input(C_t_2, cell_memory_2_read, C_o_2);

    auto out_unsqueeze_2 = tensor_iterator_2->get_iter_value(unsqueeze_o_2, -1);
    auto out_hidden_2 = tensor_iterator_2->get_iter_value(H_o_2, -1);
    auto out_cell_2 = tensor_iterator_2->get_iter_value(C_o_2, -1);

    out_hidden_2.get_tensor().set_element_type(ngPrc);
    out_cell_2.get_tensor().set_element_type(ngPrc);

    auto cell_memory_2_write = std::make_shared<Assign>(out_cell_2, var_cell_2);
    auto hidden_memory_2_write = std::make_shared<Assign>(out_hidden_2, var_hidden_2);

    auto final_reshape_pattern = std::make_shared<Constant>(element::i64, Shape{4}, std::vector<size_t>({1, 1, 1, hiddenSize}));
    auto final_reshape = std::make_shared<Reshape>(out_unsqueeze_2, final_reshape_pattern, false);

    cell_memory_write->add_control_dependency(cell_memory_read);
    hidden_memory_write->add_control_dependency(hidden_memory_read);
    cell_memory_2_write->add_control_dependency(cell_memory_2_read);
    hidden_memory_2_write->add_control_dependency(hidden_memory_2_read);

    function = std::make_shared<Function>(OutputVector {final_reshape},
                                          SinkVector{cell_memory_write, hidden_memory_write, cell_memory_2_write, hidden_memory_2_write},
                                          input_parameter,
                                          "TI_with_memory");
}

void MultipleLSTMCellTest::switchToNgraphFriendlyModel() {
    InferenceEngine::Precision netPrecision;
    std::map<std::string, std::string> config;
    size_t inputSize;
    std::tie(transformation, targetDevice, netPrecision, inputSize, hiddenSize, config) = this->GetParam();
    auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);

    std::vector<size_t> input_dims { 1, inputSize };
    std::vector<size_t> squeeze_axes {0};
    std::vector<size_t> hidden_memory_dims {1, hiddenSize};
    std::vector<size_t> cell_memory_dims {1, hiddenSize};

    auto input_parameter = builder::makeParams(ngPrc, {input_dims});

    auto input_add_const = builder::makeConstant(ngPrc, input_dims, input_bias);
    auto add = builder::makeEltwise(input_parameter[0], input_add_const, helpers::EltwiseTypes::ADD);

    auto input_mul_const = builder::makeConstant(ngPrc, input_dims, input_weights);
    auto mul = builder::makeEltwise(add, input_mul_const, helpers::EltwiseTypes::MULTIPLY);

    auto unsqueeze_input_const = std::make_shared<Constant>(element::i64, Shape{1}, squeeze_axes);
    auto unsqueeze_input = std::make_shared<Unsqueeze>(mul, unsqueeze_input_const);

    // Body 1 - layers
    auto cell_memory_constant = builder::makeConstant<float>(ngPrc, cell_memory_dims, cell_memory_init);

    auto hidden_memory_constant = builder::makeConstant<float>(ngPrc, hidden_memory_dims, hidden_memory_init);

    auto squeeze_const = std::make_shared<Constant>(element::i64, Shape{1}, squeeze_axes);
    auto squeeze = std::make_shared<Squeeze>(unsqueeze_input, squeeze_const);

    auto weightsNode = builder::makeConstant<float>(ngPrc, { 4 * hiddenSize, inputSize }, weights_vals);
    auto reccurrenceWeightsNode = builder::makeConstant<float>(ngPrc, { 4 * hiddenSize, hiddenSize }, reccurrenceWeights_vals);
    auto biasNode = builder::makeConstant<float>(ngPrc, {4 * hiddenSize}, bias_vals);
    auto lstm = std::make_shared<LSTMCell>(squeeze, hidden_memory_constant, cell_memory_constant, weightsNode,
                                                           reccurrenceWeightsNode, biasNode, hiddenSize);

    auto unsqueeze_const = std::make_shared<Constant>(element::i64, Shape{1}, squeeze_axes);
    auto unsqueeze = std::make_shared<Unsqueeze>(lstm->output(0), unsqueeze_const);

    auto first_reshape_pattern = std::make_shared<Constant>(element::i64,
                                                                        Shape{4}, std::vector<size_t>({1, 1, 1, hiddenSize}));
    auto first_reshape = std::make_shared<Reshape>(unsqueeze, first_reshape_pattern, false);
    // Body 1 - end

    auto inbetween_squeeze_const = std::make_shared<Constant>(element::i64, Shape{1}, squeeze_axes);
    auto inbetween_squeeze = std::make_shared<Squeeze>(first_reshape, inbetween_squeeze_const);

    // Body 2 - layers
    auto cell_memory_2_constant = builder::makeConstant<float>(ngPrc, cell_memory_dims, cell_memory_init);

    auto hidden_memory_2_constant = builder::makeConstant<float>(ngPrc, hidden_memory_dims, hidden_memory_init);

    auto squeeze_2_const = std::make_shared<Constant>(element::i64, Shape{1}, squeeze_axes);
    auto squeeze_2 = std::make_shared<Squeeze>(inbetween_squeeze, squeeze_2_const);

    auto weightsNode_2 = builder::makeConstant<float>(ngPrc, { 4 * hiddenSize, hiddenSize }, weights_2_vals);
    auto reccurrenceWeightsNode_2 = builder::makeConstant<float>(ngPrc, { 4 * hiddenSize, hiddenSize }, reccurrenceWeights_vals);
    auto biasNode_2 = builder::makeConstant<float>(ngPrc, {4 * hiddenSize}, bias_vals);
    auto lstm_2 = std::make_shared<LSTMCell>(squeeze_2, hidden_memory_2_constant, cell_memory_2_constant, weightsNode_2,
        reccurrenceWeightsNode_2, biasNode_2, hiddenSize);

    auto unsqueeze_2_const = std::make_shared<Constant>(element::i64, Shape{1}, squeeze_axes);
    auto unsqueeze_2 = std::make_shared<Unsqueeze>(lstm_2->output(0), unsqueeze_2_const);

    auto final_reshape_pattern = std::make_shared<Constant>(element::i64,
        Shape{4}, std::vector<size_t>({1, 1, 1, hiddenSize}));
    auto final_reshape = std::make_shared<Reshape>(unsqueeze_2, final_reshape_pattern, false);
    // Body 2 - end

    function = std::make_shared<Function>(final_reshape, input_parameter, "TI_unrolled_without_memory");
}

void MultipleLSTMCellTest::CreatePureTensorIteratorModel() {
    InferenceEngine::Precision netPrecision;
    std::map<std::string, std::string> config;
    size_t inputSize;
    std::tie(transformation, targetDevice, netPrecision, inputSize, hiddenSize, config) = this->GetParam();
    auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);

    std::vector<size_t> input_dims { 1, inputSize };
    std::vector<size_t> squeeze_axes {0};
    std::vector<size_t> hidden_memory_dims {1, hiddenSize};
    std::vector<size_t> cell_memory_dims {1, hiddenSize};

    auto input_parameter = builder::makeParams(ngPrc, {input_dims});

    auto input_add_const = builder::makeConstant(ngPrc, input_dims, input_bias);
    auto add = builder::makeEltwise(input_parameter[0], input_add_const, helpers::EltwiseTypes::ADD);

    auto input_mul_const = builder::makeConstant(ngPrc, input_dims, input_weights);
    auto mul = builder::makeEltwise(add, input_mul_const, helpers::EltwiseTypes::MULTIPLY);

    auto unsqueeze_input_const = std::make_shared<Constant>(element::i64, Shape{1}, squeeze_axes);
    auto unsqueeze_input = std::make_shared<Unsqueeze>(mul, unsqueeze_input_const);

    auto permute_in_params = std::make_shared<Constant>(element::i64, Shape{3}, Shape{{1, 0, 2}});
    auto permute_in = std::make_shared<Transpose>(unsqueeze_input, permute_in_params);

    auto cell_memory_constant = builder::makeConstant<float>(ngPrc, cell_memory_dims, cell_memory_init);

    auto hidden_memory_constant = builder::makeConstant<float>(ngPrc, hidden_memory_dims, hidden_memory_init);

    // Body - inputs
    auto X = std::make_shared<Parameter>(ngPrc, Shape{1, 1, inputSize});
    auto H_t = std::make_shared<Parameter>(ngPrc, Shape{1, hiddenSize});
    auto C_t = std::make_shared<Parameter>(ngPrc, Shape{1, hiddenSize});
    H_t->set_friendly_name("hidden_state_1");
    C_t->set_friendly_name("cell_state_1");
    // Body - layers
    auto squeeze_const = std::make_shared<Constant>(element::i64, Shape{1}, squeeze_axes);
    auto squeeze = std::make_shared<Squeeze>(X, squeeze_const);

    auto weightsNode = builder::makeConstant<float>(ngPrc, { 4 * hiddenSize, inputSize }, weights_vals);
    auto reccurrenceWeightsNode = builder::makeConstant<float>(ngPrc, { 4 * hiddenSize, hiddenSize }, reccurrenceWeights_vals);
    auto biasNode = builder::makeConstant<float>(ngPrc, {4 * hiddenSize}, bias_vals);
    auto lstm = std::make_shared<LSTMCell>(squeeze, H_t, C_t, weightsNode, reccurrenceWeightsNode, biasNode, hiddenSize);

    auto unsqueeze_const = std::make_shared<Constant>(element::i64, Shape{1}, squeeze_axes);
    auto unsqueeze = std::make_shared<Unsqueeze>(lstm->output(0), unsqueeze_const);
    // body - outputs
    auto H_o = lstm->output(0);
    auto C_o = lstm->output(1);
    auto unsqueeze_o = unsqueeze->output(0);

    auto body = std::make_shared<Function>(OutputVector{unsqueeze_o, H_o, C_o}, ParameterVector {X, H_t, C_t});
    // TI construction
    auto tensor_iterator = std::make_shared<TensorIterator>();
    tensor_iterator->set_body(body);
    tensor_iterator->set_sliced_input(X, permute_in, 0, 1, 1, -1, 0);
    tensor_iterator->set_merged_input(H_t, hidden_memory_constant, H_o);
    tensor_iterator->set_merged_input(C_t, cell_memory_constant, C_o);

    auto out_unsqueeze = tensor_iterator->get_iter_value(unsqueeze_o, -1);
    auto out_hidden = tensor_iterator->get_iter_value(H_o, -1);
    auto out_cell = tensor_iterator->get_iter_value(C_o, -1);

    out_hidden.get_tensor().set_element_type(ngPrc);
    out_cell.get_tensor().set_element_type(ngPrc);
    tensor_iterator->validate_and_infer_types();

    auto first_reshape_pattern = std::make_shared<Constant>(element::i64,
                                                                        Shape{4}, std::vector<size_t>({1, 1, 1, hiddenSize}));
    auto first_reshape = std::make_shared<Reshape>(out_unsqueeze, first_reshape_pattern, false);
    // End of TI 1

    auto inbetween_squeeze_const = std::make_shared<Constant>(element::i64, Shape{1}, squeeze_axes);
    auto inbetween_squeeze = std::make_shared<Squeeze>(first_reshape, inbetween_squeeze_const);

    // Second TI
    auto cell_memory_2_constant = builder::makeConstant<float>(ngPrc, cell_memory_dims, cell_memory_init);

    auto hidden_memory_2_constant = builder::makeConstant<float>(ngPrc, hidden_memory_dims, hidden_memory_init);

    // Body - inputs
    auto X_2 = std::make_shared<Parameter>(ngPrc, Shape{1, 1, hiddenSize});
    auto H_t_2 = std::make_shared<Parameter>(ngPrc, Shape{1, hiddenSize});
    auto C_t_2 = std::make_shared<Parameter>(ngPrc, Shape{1, hiddenSize});
    H_t_2->set_friendly_name("hidden_state_2");
    C_t_2->set_friendly_name("cell_state_2");
    // Body - layers
    auto squeeze_2_const = std::make_shared<Constant>(element::i64, Shape{1}, squeeze_axes);
    auto squeeze_2 = std::make_shared<Squeeze>(X_2, squeeze_2_const);

    auto weightsNode_2 = builder::makeConstant<float>(ngPrc, { 4 * hiddenSize, hiddenSize }, weights_2_vals);
    auto reccurrenceWeightsNode_2 = builder::makeConstant<float>(ngPrc, { 4 * hiddenSize, hiddenSize }, reccurrenceWeights_vals);
    auto biasNode_2 = builder::makeConstant<float>(ngPrc, {4 * hiddenSize}, bias_vals);
    auto lstm_2 = std::make_shared<LSTMCell>(squeeze_2, H_t_2, C_t_2, weightsNode_2, reccurrenceWeightsNode_2, biasNode_2, hiddenSize);

    auto unsqueeze_2_const = std::make_shared<Constant>(element::i64, Shape{1}, squeeze_axes);
    auto unsqueeze_2 = std::make_shared<Unsqueeze>(lstm_2->output(0), unsqueeze_2_const);
    // body - outputs
    auto H_o_2 = lstm_2->output(0);
    auto C_o_2 = lstm_2->output(1);
    auto unsqueeze_o_2 = unsqueeze_2->output(0);

    auto body_2 = std::make_shared<Function>(OutputVector{unsqueeze_o_2, H_o_2, C_o_2}, ParameterVector {X_2, H_t_2, C_t_2});
    // TI construction
    auto tensor_iterator_2 = std::make_shared<TensorIterator>();
    tensor_iterator_2->set_body(body_2);
    tensor_iterator_2->set_sliced_input(X_2, inbetween_squeeze, 0, 1, 1, -1, 0);
    tensor_iterator_2->set_merged_input(H_t_2, hidden_memory_2_constant, H_o_2);
    tensor_iterator_2->set_merged_input(C_t_2, cell_memory_2_constant, C_o_2);

    auto out_unsqueeze_2 = tensor_iterator_2->get_iter_value(unsqueeze_o_2, -1);
    auto out_hidden_2 = tensor_iterator_2->get_iter_value(H_o_2, -1);
    auto out_cell_2 = tensor_iterator_2->get_iter_value(C_o_2, -1);

    out_hidden_2.get_tensor().set_element_type(ngPrc);
    out_cell_2.get_tensor().set_element_type(ngPrc);
    tensor_iterator_2->validate_and_infer_types();
    auto final_reshape_pattern = std::make_shared<Constant>(element::i64,
                                                                        Shape{4}, std::vector<size_t>({1, 1, 1, hiddenSize}));
    auto final_reshape = std::make_shared<Reshape>(out_unsqueeze_2, final_reshape_pattern, false);

    function = std::make_shared<Function>(final_reshape, input_parameter, "PureTI");
}

void MultipleLSTMCellTest::InitMemory() {
    InferenceEngine::TensorDesc state_description(InferenceEngine::Precision::FP32,
                                                  InferenceEngine::SizeVector({1, hiddenSize}),
                                                  InferenceEngine::Layout::NC);
    IE_SUPPRESS_DEPRECATED_START
    auto states = executableNetwork.QueryState();
    for (auto& state : states) {
        auto name = state.GetName();
        if (name.find("cell_state_1") != std::string::npos) {
            auto blob = FuncTestUtils::createAndFillBlobWithFloatArray(state_description,
                                                                       cell_memory_init.data(), cell_memory_init.size());
            state.SetState(blob);
        } else if (name.find("hidden_state_1") != std::string::npos) {
            auto blob = FuncTestUtils::createAndFillBlobWithFloatArray(state_description,
                                                                       hidden_memory_init.data(), hidden_memory_init.size());
            state.SetState(blob);
        } else if (name.find("cell_state_2") != std::string::npos) {
            auto blob = FuncTestUtils::createAndFillBlobWithFloatArray(state_description,
                                                                       cell_memory_init.data(), cell_memory_init.size());
            state.SetState(blob);
        } else if (name.find("hidden_state_2") != std::string::npos) {
            auto blob = FuncTestUtils::createAndFillBlobWithFloatArray(state_description,
                                                                       hidden_memory_init.data(), hidden_memory_init.size());
            state.SetState(blob);
        } else {
            GTEST_FAIL() << "unknown memory state";
        }
    }
    IE_SUPPRESS_DEPRECATED_END
}

void MultipleLSTMCellTest::ApplyLowLatency() {
    // Calculate values after LowLatency transformation
    CreatePureTensorIteratorModel();
    if (transformation == ngraph::helpers::MemoryTransformation::LOW_LATENCY) {
        function->validate_nodes_and_infer_types();
        // Apply LowLatency (insert Assigns/ReadValues) and UnrollTensorIterator
        pass::Manager manager;
        NGRAPH_SUPPRESS_DEPRECATED_START
        manager.register_pass<ngraph::pass::LowLatency>();
        NGRAPH_SUPPRESS_DEPRECATED_END // LowLatency enables UnrollTI
        manager.run_passes(function);
        bool ti_found = helpers::is_tensor_iterator_exist(function);
        EXPECT_EQ(ti_found, true);
        LoadNetwork();
    } else if (transformation == ngraph::helpers::MemoryTransformation::LOW_LATENCY_V2) {
        function->validate_nodes_and_infer_types();
        // Apply LowLatency (insert Assigns/ReadValues) and UnrollTensorIterator

        pass::Manager manager;
        manager.register_pass<pass::LowLatency2>();
        manager.run_passes(function);
        bool ti_found = helpers::is_tensor_iterator_exist(function);
        EXPECT_EQ(ti_found, false);
        LoadNetwork();
    } else if (transformation == ngraph::helpers::MemoryTransformation::LOW_LATENCY_REGULAR_API) {
        cnnNetwork = InferenceEngine::CNNNetwork{function};
        IE_SUPPRESS_DEPRECATED_START
        InferenceEngine::LowLatency(cnnNetwork);
        IE_SUPPRESS_DEPRECATED_END

        bool ti_found = helpers::is_tensor_iterator_exist(cnnNetwork.getFunction());
        EXPECT_EQ(ti_found, true);

        ConfigureNetwork();
        executableNetwork = core->LoadNetwork(cnnNetwork, targetDevice, configuration);
    } else if (transformation == ngraph::helpers::MemoryTransformation::LOW_LATENCY_V2_REGULAR_API) {
        cnnNetwork = InferenceEngine::CNNNetwork{function};
        InferenceEngine::lowLatency2(cnnNetwork);

        bool ti_found = helpers::is_tensor_iterator_exist(cnnNetwork.getFunction());
        EXPECT_EQ(ti_found, false);

        ConfigureNetwork();
        executableNetwork = core->LoadNetwork(cnnNetwork, targetDevice, configuration);
    }
}

void MultipleLSTMCellTest::Run() {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    if (transformation != ngraph::helpers::MemoryTransformation::NONE) {
        ApplyLowLatency();
    } else {
        LoadNetwork();
    }

    InitMemory();
    GenerateInputs();
    Infer();

    // Calculate ref values
    if (transformation == ngraph::helpers::MemoryTransformation::NONE) {
        switchToNgraphFriendlyModel();
    } else {
        CreatePureTensorIteratorModel();
    }
    Validate();
}
}  // namespace SubgraphTestsDefinitions
