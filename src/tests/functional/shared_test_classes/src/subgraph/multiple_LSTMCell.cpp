// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/subgraph/multiple_LSTMCell.hpp"
#include "common_test_utils/node_builders/eltwise.hpp"
#include "common_test_utils/data_utils.hpp"
#include "common_test_utils/ov_test_utils.hpp"
#include "functional_test_utils/skip_tests_config.hpp"
#include "openvino/pass/low_latency.hpp"
#include "openvino/pass/manager.hpp"
#include "common_test_utils/node_builders/constant.hpp"

namespace ov {
namespace test {

std::string MultipleLSTMCellTest::getTestCaseName(const testing::TestParamInfo<multipleLSTMCellParams>& obj) {
    std::string targetDevice;
    ov::element::Type element_type;
    size_t inputSize;
    size_t hiddenSize;
    ov::AnyMap config;
    ov::test::utils::MemoryTransformation transformation;
    std::tie(transformation, targetDevice, element_type, inputSize, hiddenSize, config) = obj.param;
    std::ostringstream result;

    result << "transformation=" << transformation << "_";
    result << "ET=" << element_type << "_";
    result << "IS=" << inputSize << "_";
    result << "HS=" << hiddenSize << "_";
    result << "targetDevice=" << targetDevice;
    return result.str();
}

void MultipleLSTMCellTest::SetUp() {
    ov::element::Type element_type;
    ov::AnyMap config;
    size_t inputSize;
    std::tie(transformation, targetDevice, element_type, inputSize, hiddenSize, config) = this->GetParam();
    configuration.insert(config.begin(), config.end());

    std::vector<size_t> input_dims{1, inputSize};
    std::vector<size_t> squeeze_axes{0};
    std::vector<size_t> hidden_memory_dims{1, hiddenSize};
    std::vector<size_t> cell_memory_dims{1, hiddenSize};
    input_shapes = {input_dims};

    input_bias = ov::test::utils::generate_float_numbers(inputSize, -0.25f, 0.0f);
    input_weights = ov::test::utils::generate_float_numbers(inputSize, 0.0f, 0.15f);
    hidden_memory_init = ov::test::utils::generate_float_numbers(hiddenSize, -0.2f, 0.2f);
    cell_memory_init = ov::test::utils::generate_float_numbers(hiddenSize, -0.2f, 0.2f);
    weights_vals = ov::test::utils::generate_float_numbers(4 * hiddenSize * inputSize, -0.1f, 0.1f);
    weights_2_vals = ov::test::utils::generate_float_numbers(4 * hiddenSize * hiddenSize, -0.1f, 0.1f);
    reccurrenceWeights_vals = ov::test::utils::generate_float_numbers(4 * hiddenSize * hiddenSize, -0.1f, 0.1f);
    bias_vals = ov::test::utils::generate_float_numbers(4 * hiddenSize, -0.25f, 0.15f);

    ov::ParameterVector input_parameter{std::make_shared<ov::op::v0::Parameter>(element_type, ov::Shape(input_dims))};
    input_parameter[0]->set_friendly_name("Parameter_1");

    auto input_add_const = ov::op::v0::Constant::create(element_type, input_dims, input_bias);
    auto add = ov::test::utils::make_eltwise(input_parameter[0], input_add_const, ov::test::utils::EltwiseTypes::ADD);

    auto input_mul_const = ov::op::v0::Constant::create(element_type, input_dims, input_weights);
    auto mul = ov::test::utils::make_eltwise(add, input_mul_const, ov::test::utils::EltwiseTypes::MULTIPLY);

    auto unsqueeze_input_const = std::make_shared<ov::op::v0::Constant>(element::i64, Shape{1}, squeeze_axes);
    auto unsqueeze_input = std::make_shared<ov::op::v0::Unsqueeze>(mul, unsqueeze_input_const);

    auto permute_in_params = std::make_shared<ov::op::v0::Constant>(element::i64, Shape{3}, Shape{{1, 0, 2}});
    auto permute_in = std::make_shared<ov::op::v1::Transpose>(unsqueeze_input, permute_in_params);

    auto cell_memory_constant = ov::op::v0::Constant::create(element_type, cell_memory_dims, cell_memory_init);
    auto var_cell = std::make_shared<ov::op::util::Variable>(
        ov::op::util::VariableInfo{PartialShape::dynamic(), element::dynamic, "cell_state_1"});
    auto var_hidden = std::make_shared<ov::op::util::Variable>(
        ov::op::util::VariableInfo{PartialShape::dynamic(), element::dynamic, "hidden_state_1"});
    auto cell_memory_read = std::make_shared<ov::op::v6::ReadValue>(cell_memory_constant, var_cell);
    cell_memory_read->set_friendly_name("cell_memory");

    auto hidden_memory_constant =
        ov::op::v0::Constant::create(element_type, hidden_memory_dims, hidden_memory_init);
    auto hidden_memory_read = std::make_shared<ov::op::v6::ReadValue>(hidden_memory_constant, var_hidden);
    hidden_memory_read->set_friendly_name("hidden_memory");

    // Body - inputs
    auto X = std::make_shared<ov::op::v0::Parameter>(element_type, Shape{1, 1, inputSize});
    auto H_t = std::make_shared<ov::op::v0::Parameter>(element_type, Shape{1, hiddenSize});
    auto C_t = std::make_shared<ov::op::v0::Parameter>(element_type, Shape{1, hiddenSize});
    // Body - layers
    auto squeeze_const = std::make_shared<ov::op::v0::Constant>(element::i64, Shape{1}, squeeze_axes);
    auto squeeze = std::make_shared<ov::op::v0::Squeeze>(X, squeeze_const);

    auto weightsNode = ov::op::v0::Constant::create(element_type, {4 * hiddenSize, inputSize}, weights_vals);
    auto reccurrenceWeightsNode =
        ov::op::v0::Constant::create(element_type, {4 * hiddenSize, hiddenSize}, reccurrenceWeights_vals);
    auto biasNode = ov::op::v0::Constant::create(element_type, {4 * hiddenSize}, bias_vals);
    auto lstm = std::make_shared<ov::op::v0::LSTMCell>(squeeze,
                                                       H_t,
                                                       C_t,
                                                       weightsNode,
                                                       reccurrenceWeightsNode,
                                                       biasNode,
                                                       hiddenSize);

    auto unsqueeze_const = std::make_shared<ov::op::v0::Constant>(element::i64, Shape{1}, squeeze_axes);
    auto unsqueeze = std::make_shared<ov::op::v0::Unsqueeze>(lstm->output(0), unsqueeze_const);
    // body - outputs
    auto H_o = lstm->output(0);
    auto C_o = lstm->output(1);
    auto unsqueeze_o = unsqueeze->output(0);

    auto body = std::make_shared<Model>(OutputVector{unsqueeze_o, H_o, C_o}, ParameterVector{X, H_t, C_t});
    // TI construction
    auto tensor_iterator = std::make_shared<ov::op::v0::TensorIterator>();
    tensor_iterator->set_body(body);
    tensor_iterator->set_sliced_input(X, permute_in, 0, 1, 1, -1, 0);
    tensor_iterator->set_merged_input(H_t, hidden_memory_read, H_o);
    tensor_iterator->set_merged_input(C_t, cell_memory_read, C_o);
    tensor_iterator->validate_and_infer_types();

    auto out_unsqueeze = tensor_iterator->get_iter_value(unsqueeze_o, -1);
    auto out_hidden = tensor_iterator->get_iter_value(H_o, -1);
    auto out_cell = tensor_iterator->get_iter_value(C_o, -1);

    auto cell_memory_write = std::make_shared<ov::op::v6::Assign>(out_cell, var_cell);
    auto hidden_memory_write = std::make_shared<ov::op::v6::Assign>(out_hidden, var_hidden);

    auto first_reshape_pattern =
        std::make_shared<ov::op::v0::Constant>(element::i64, Shape{4}, std::vector<size_t>({1, 1, 1, hiddenSize}));
    auto first_reshape = std::make_shared<ov::op::v1::Reshape>(out_unsqueeze, first_reshape_pattern, false);
    // End of TI 1

    auto inbetween_squeeze_const = std::make_shared<ov::op::v0::Constant>(element::i64, Shape{1}, squeeze_axes);
    auto inbetween_squeeze = std::make_shared<ov::op::v0::Squeeze>(first_reshape, inbetween_squeeze_const);

    // Second TI
    auto var_cell_2 = std::make_shared<ov::op::util::Variable>(
        ov::op::util::VariableInfo{PartialShape::dynamic(), element::dynamic, "cell_state_2"});
    auto var_hidden_2 = std::make_shared<ov::op::util::Variable>(
        ov::op::util::VariableInfo{PartialShape::dynamic(), element::dynamic, "hidden_state_2"});
    auto cell_memory_2_constant =
        ov::op::v0::Constant::create(element_type, cell_memory_dims, cell_memory_init);
    auto cell_memory_2_read = std::make_shared<ov::op::v6::ReadValue>(cell_memory_2_constant, var_cell_2);
    cell_memory_2_read->set_friendly_name("cell_memory_2");

    auto hidden_memory_2_constant =
        ov::op::v0::Constant::create(element_type, hidden_memory_dims, hidden_memory_init);
    auto hidden_memory_2_read = std::make_shared<ov::op::v6::ReadValue>(hidden_memory_2_constant, var_hidden_2);
    hidden_memory_2_read->set_friendly_name("hidden_memory_2");

    // Body - inputs
    auto X_2 = std::make_shared<ov::op::v0::Parameter>(element_type, Shape{1, 1, hiddenSize});
    auto H_t_2 = std::make_shared<ov::op::v0::Parameter>(element_type, Shape{1, hiddenSize});
    auto C_t_2 = std::make_shared<ov::op::v0::Parameter>(element_type, Shape{1, hiddenSize});
    // Body - layers
    auto squeeze_2_const = std::make_shared<ov::op::v0::Constant>(element::i64, Shape{1}, squeeze_axes);
    auto squeeze_2 = std::make_shared<ov::op::v0::Squeeze>(X_2, squeeze_2_const);

    auto weightsNode_2 =
        ov::op::v0::Constant::create(element_type, {4 * hiddenSize, hiddenSize}, weights_2_vals);
    auto reccurrenceWeightsNode_2 =
        ov::op::v0::Constant::create(element_type, {4 * hiddenSize, hiddenSize}, reccurrenceWeights_vals);
    auto biasNode_2 = ov::op::v0::Constant::create(element_type, {4 * hiddenSize}, bias_vals);
    auto lstm_2 = std::make_shared<ov::op::v0::LSTMCell>(squeeze_2,
                                                         H_t_2,
                                                         C_t_2,
                                                         weightsNode_2,
                                                         reccurrenceWeightsNode_2,
                                                         biasNode_2,
                                                         hiddenSize);

    auto unsqueeze_2_const = std::make_shared<ov::op::v0::Constant>(element::i64, Shape{1}, squeeze_axes);
    auto unsqueeze_2 = std::make_shared<ov::op::v0::Unsqueeze>(lstm_2->output(0), unsqueeze_2_const);
    // body - outputs
    auto H_o_2 = lstm_2->output(0);
    auto C_o_2 = lstm_2->output(1);
    auto unsqueeze_o_2 = unsqueeze_2->output(0);

    auto body_2 =
        std::make_shared<Model>(OutputVector{unsqueeze_o_2, H_o_2, C_o_2}, ParameterVector{X_2, H_t_2, C_t_2});
    // TI construction
    auto tensor_iterator_2 = std::make_shared<ov::op::v0::TensorIterator>();
    tensor_iterator_2->set_body(body_2);
    tensor_iterator_2->set_sliced_input(X_2, inbetween_squeeze, 0, 1, 1, -1, 0);
    tensor_iterator_2->set_merged_input(H_t_2, hidden_memory_2_read, H_o_2);
    tensor_iterator_2->set_merged_input(C_t_2, cell_memory_2_read, C_o_2);
    tensor_iterator_2->validate_and_infer_types();

    auto out_unsqueeze_2 = tensor_iterator_2->get_iter_value(unsqueeze_o_2, -1);
    auto out_hidden_2 = tensor_iterator_2->get_iter_value(H_o_2, -1);
    auto out_cell_2 = tensor_iterator_2->get_iter_value(C_o_2, -1);

    auto cell_memory_2_write = std::make_shared<ov::op::v6::Assign>(out_cell_2, var_cell_2);
    auto hidden_memory_2_write = std::make_shared<ov::op::v6::Assign>(out_hidden_2, var_hidden_2);

    auto final_reshape_pattern =
        std::make_shared<ov::op::v0::Constant>(element::i64, Shape{4}, std::vector<size_t>({1, 1, 1, hiddenSize}));
    auto final_reshape = std::make_shared<ov::op::v1::Reshape>(out_unsqueeze_2, final_reshape_pattern, false);
    final_reshape->set_friendly_name("Reshape_1");

    cell_memory_write->add_control_dependency(cell_memory_read);
    hidden_memory_write->add_control_dependency(hidden_memory_read);
    cell_memory_2_write->add_control_dependency(cell_memory_2_read);
    hidden_memory_2_write->add_control_dependency(hidden_memory_2_read);

    function = std::make_shared<Model>(
        OutputVector{final_reshape},
        SinkVector{cell_memory_write, hidden_memory_write, cell_memory_2_write, hidden_memory_2_write},
        input_parameter,
        "TI_with_memory");
}

void MultipleLSTMCellTest::switch_to_friendly_model() {
    ov::element::Type element_type;
    ov::AnyMap config;
    size_t inputSize;
    std::tie(transformation, targetDevice, element_type, inputSize, hiddenSize, config) = this->GetParam();

    std::vector<size_t> input_dims{1, inputSize};
    std::vector<size_t> squeeze_axes{0};
    std::vector<size_t> hidden_memory_dims{1, hiddenSize};
    std::vector<size_t> cell_memory_dims{1, hiddenSize};

    ov::ParameterVector input_parameter{std::make_shared<ov::op::v0::Parameter>(element_type, ov::Shape(input_dims))};
    input_parameter[0]->set_friendly_name("Parameter_1");

    auto input_add_const = ov::op::v0::Constant::create(element_type, input_dims, input_bias);
    auto add = ov::test::utils::make_eltwise(input_parameter[0], input_add_const, ov::test::utils::EltwiseTypes::ADD);

    auto input_mul_const = ov::op::v0::Constant::create(element_type, input_dims, input_weights);
    auto mul = ov::test::utils::make_eltwise(add, input_mul_const, ov::test::utils::EltwiseTypes::MULTIPLY);

    auto unsqueeze_input_const = std::make_shared<ov::op::v0::Constant>(element::i64, Shape{1}, squeeze_axes);
    auto unsqueeze_input = std::make_shared<ov::op::v0::Unsqueeze>(mul, unsqueeze_input_const);

    // Body 1 - layers
    auto cell_memory_constant = ov::op::v0::Constant::create(element_type, cell_memory_dims, cell_memory_init);

    auto hidden_memory_constant =
        ov::op::v0::Constant::create(element_type, hidden_memory_dims, hidden_memory_init);

    auto squeeze_const = std::make_shared<ov::op::v0::Constant>(element::i64, Shape{1}, squeeze_axes);
    auto squeeze = std::make_shared<ov::op::v0::Squeeze>(unsqueeze_input, squeeze_const);

    auto weightsNode = ov::op::v0::Constant::create(element_type, {4 * hiddenSize, inputSize}, weights_vals);
    auto reccurrenceWeightsNode =
        ov::op::v0::Constant::create(element_type, {4 * hiddenSize, hiddenSize}, reccurrenceWeights_vals);
    auto biasNode = ov::op::v0::Constant::create(element_type, {4 * hiddenSize}, bias_vals);
    auto lstm = std::make_shared<ov::op::v0::LSTMCell>(squeeze,
                                                       hidden_memory_constant,
                                                       cell_memory_constant,
                                                       weightsNode,
                                                       reccurrenceWeightsNode,
                                                       biasNode,
                                                       hiddenSize,
                                                       ov::op::LSTMWeightsFormat::FICO);

    auto unsqueeze_const = std::make_shared<ov::op::v0::Constant>(element::i64, Shape{1}, squeeze_axes);
    auto unsqueeze = std::make_shared<ov::op::v0::Unsqueeze>(lstm->output(0), unsqueeze_const);

    auto first_reshape_pattern =
        std::make_shared<ov::op::v0::Constant>(element::i64, Shape{4}, std::vector<size_t>({1, 1, 1, hiddenSize}));
    auto first_reshape = std::make_shared<ov::op::v1::Reshape>(unsqueeze, first_reshape_pattern, false);
    // Body 1 - end

    auto inbetween_squeeze_const = std::make_shared<ov::op::v0::Constant>(element::i64, Shape{1}, squeeze_axes);
    auto inbetween_squeeze = std::make_shared<ov::op::v0::Squeeze>(first_reshape, inbetween_squeeze_const);

    // Body 2 - layers
    auto cell_memory_2_constant =
        ov::op::v0::Constant::create(element_type, cell_memory_dims, cell_memory_init);

    auto hidden_memory_2_constant =
        ov::op::v0::Constant::create(element_type, hidden_memory_dims, hidden_memory_init);

    auto squeeze_2_const = std::make_shared<ov::op::v0::Constant>(element::i64, Shape{1}, squeeze_axes);
    auto squeeze_2 = std::make_shared<ov::op::v0::Squeeze>(inbetween_squeeze, squeeze_2_const);

    auto weightsNode_2 =
        ov::op::v0::Constant::create(element_type, {4 * hiddenSize, hiddenSize}, weights_2_vals);
    auto reccurrenceWeightsNode_2 =
        ov::op::v0::Constant::create(element_type, {4 * hiddenSize, hiddenSize}, reccurrenceWeights_vals);
    auto biasNode_2 = ov::op::v0::Constant::create(element_type, {4 * hiddenSize}, bias_vals);
    auto lstm_2 = std::make_shared<ov::op::v0::LSTMCell>(squeeze_2,
                                                         hidden_memory_2_constant,
                                                         cell_memory_2_constant,
                                                         weightsNode_2,
                                                         reccurrenceWeightsNode_2,
                                                         biasNode_2,
                                                         hiddenSize,
                                                         ov::op::LSTMWeightsFormat::FICO);

    auto unsqueeze_2_const = std::make_shared<ov::op::v0::Constant>(element::i64, Shape{1}, squeeze_axes);
    auto unsqueeze_2 = std::make_shared<ov::op::v0::Unsqueeze>(lstm_2->output(0), unsqueeze_2_const);

    auto final_reshape_pattern =
        std::make_shared<ov::op::v0::Constant>(element::i64, Shape{4}, std::vector<size_t>({1, 1, 1, hiddenSize}));
    auto final_reshape = std::make_shared<ov::op::v1::Reshape>(unsqueeze_2, final_reshape_pattern, false);
    final_reshape->set_friendly_name("Reshape_1");
    // Body 2 - end

    functionRefs = std::make_shared<Model>(final_reshape, input_parameter, "TI_unrolled_without_memory");
}

void MultipleLSTMCellTest::create_pure_tensor_iterator_model() {
    ov::element::Type element_type;
    ov::AnyMap config;
    size_t inputSize;
    std::tie(transformation, targetDevice, element_type, inputSize, hiddenSize, config) = this->GetParam();

    std::vector<size_t> input_dims{1, inputSize};
    std::vector<size_t> squeeze_axes{0};
    std::vector<size_t> hidden_memory_dims{1, hiddenSize};
    std::vector<size_t> cell_memory_dims{1, hiddenSize};

    ov::ParameterVector input_parameter{std::make_shared<ov::op::v0::Parameter>(element_type, ov::Shape(input_dims))};
    input_parameter[0]->set_friendly_name("Parameter_1");

    auto input_add_const = ov::op::v0::Constant::create(element_type, input_dims, input_bias);
    auto add = ov::test::utils::make_eltwise(input_parameter[0], input_add_const, ov::test::utils::EltwiseTypes::ADD);

    auto input_mul_const = ov::op::v0::Constant::create(element_type, input_dims, input_weights);
    auto mul = ov::test::utils::make_eltwise(add, input_mul_const, ov::test::utils::EltwiseTypes::MULTIPLY);

    auto unsqueeze_input_const = std::make_shared<ov::op::v0::Constant>(element::i64, Shape{1}, squeeze_axes);
    auto unsqueeze_input = std::make_shared<ov::op::v0::Unsqueeze>(mul, unsqueeze_input_const);

    auto permute_in_params = std::make_shared<ov::op::v0::Constant>(element::i64, Shape{3}, Shape{{1, 0, 2}});
    auto permute_in = std::make_shared<ov::op::v1::Transpose>(unsqueeze_input, permute_in_params);

    auto cell_memory_constant = ov::op::v0::Constant::create(element_type, cell_memory_dims, cell_memory_init);

    auto hidden_memory_constant =
        ov::op::v0::Constant::create(element_type, hidden_memory_dims, hidden_memory_init);

    // Body - inputs
    auto X = std::make_shared<ov::op::v0::Parameter>(element_type, Shape{1, 1, inputSize});
    auto H_t = std::make_shared<ov::op::v0::Parameter>(element_type, Shape{1, hiddenSize});
    auto C_t = std::make_shared<ov::op::v0::Parameter>(element_type, Shape{1, hiddenSize});
    H_t->set_friendly_name("hidden_state_1");
    C_t->set_friendly_name("cell_state_1");
    // Body - layers
    auto squeeze_const = std::make_shared<ov::op::v0::Constant>(element::i64, Shape{1}, squeeze_axes);
    auto squeeze = std::make_shared<ov::op::v0::Squeeze>(X, squeeze_const);

    auto weightsNode = ov::op::v0::Constant::create(element_type, {4 * hiddenSize, inputSize}, weights_vals);
    auto reccurrenceWeightsNode =
        ov::op::v0::Constant::create(element_type, {4 * hiddenSize, hiddenSize}, reccurrenceWeights_vals);
    auto biasNode = ov::op::v0::Constant::create(element_type, {4 * hiddenSize}, bias_vals);
    auto lstm = std::make_shared<ov::op::v0::LSTMCell>(squeeze,
                                                       H_t,
                                                       C_t,
                                                       weightsNode,
                                                       reccurrenceWeightsNode,
                                                       biasNode,
                                                       hiddenSize,
                                                       ov::op::LSTMWeightsFormat::FICO);

    auto unsqueeze_const = std::make_shared<ov::op::v0::Constant>(element::i64, Shape{1}, squeeze_axes);
    auto unsqueeze = std::make_shared<ov::op::v0::Unsqueeze>(lstm->output(0), unsqueeze_const);
    // body - outputs
    auto H_o = lstm->output(0);
    auto C_o = lstm->output(1);
    auto unsqueeze_o = unsqueeze->output(0);

    auto body = std::make_shared<Model>(OutputVector{unsqueeze_o, H_o, C_o}, ParameterVector{X, H_t, C_t});
    // TI construction
    auto tensor_iterator = std::make_shared<ov::op::v0::TensorIterator>();
    tensor_iterator->set_body(body);
    tensor_iterator->set_sliced_input(X, permute_in, 0, 1, 1, -1, 0);
    tensor_iterator->set_merged_input(H_t, hidden_memory_constant, H_o);
    tensor_iterator->set_merged_input(C_t, cell_memory_constant, C_o);

    auto out_unsqueeze = tensor_iterator->get_iter_value(unsqueeze_o, -1);
    auto out_hidden = tensor_iterator->get_iter_value(H_o, -1);
    auto out_cell = tensor_iterator->get_iter_value(C_o, -1);

    tensor_iterator->validate_and_infer_types();

    auto first_reshape_pattern =
        std::make_shared<ov::op::v0::Constant>(element::i64, Shape{4}, std::vector<size_t>({1, 1, 1, hiddenSize}));
    auto first_reshape = std::make_shared<ov::op::v1::Reshape>(out_unsqueeze, first_reshape_pattern, false);
    // End of TI 1

    auto inbetween_squeeze_const = std::make_shared<ov::op::v0::Constant>(element::i64, Shape{1}, squeeze_axes);
    auto inbetween_squeeze = std::make_shared<ov::op::v0::Squeeze>(first_reshape, inbetween_squeeze_const);

    // Second TI
    auto cell_memory_2_constant =
        ov::op::v0::Constant::create(element_type, cell_memory_dims, cell_memory_init);

    auto hidden_memory_2_constant =
        ov::op::v0::Constant::create(element_type, hidden_memory_dims, hidden_memory_init);

    // Body - inputs
    auto X_2 = std::make_shared<ov::op::v0::Parameter>(element_type, Shape{1, 1, hiddenSize});
    auto H_t_2 = std::make_shared<ov::op::v0::Parameter>(element_type, Shape{1, hiddenSize});
    auto C_t_2 = std::make_shared<ov::op::v0::Parameter>(element_type, Shape{1, hiddenSize});
    H_t_2->set_friendly_name("hidden_state_2");
    C_t_2->set_friendly_name("cell_state_2");
    // Body - layers
    auto squeeze_2_const = std::make_shared<ov::op::v0::Constant>(element::i64, Shape{1}, squeeze_axes);
    auto squeeze_2 = std::make_shared<ov::op::v0::Squeeze>(X_2, squeeze_2_const);

    auto weightsNode_2 =
        ov::op::v0::Constant::create(element_type, {4 * hiddenSize, hiddenSize}, weights_2_vals);
    auto reccurrenceWeightsNode_2 =
        ov::op::v0::Constant::create(element_type, {4 * hiddenSize, hiddenSize}, reccurrenceWeights_vals);
    auto biasNode_2 = ov::op::v0::Constant::create(element_type, {4 * hiddenSize}, bias_vals);
    auto lstm_2 = std::make_shared<ov::op::v0::LSTMCell>(squeeze_2,
                                                         H_t_2,
                                                         C_t_2,
                                                         weightsNode_2,
                                                         reccurrenceWeightsNode_2,
                                                         biasNode_2,
                                                         hiddenSize,
                                                         ov::op::LSTMWeightsFormat::FICO);

    auto unsqueeze_2_const = std::make_shared<ov::op::v0::Constant>(element::i64, Shape{1}, squeeze_axes);
    auto unsqueeze_2 = std::make_shared<ov::op::v0::Unsqueeze>(lstm_2->output(0), unsqueeze_2_const);
    // body - outputs
    auto H_o_2 = lstm_2->output(0);
    auto C_o_2 = lstm_2->output(1);
    auto unsqueeze_o_2 = unsqueeze_2->output(0);

    auto body_2 =
        std::make_shared<Model>(OutputVector{unsqueeze_o_2, H_o_2, C_o_2}, ParameterVector{X_2, H_t_2, C_t_2});
    // TI construction
    auto tensor_iterator_2 = std::make_shared<ov::op::v0::TensorIterator>();
    tensor_iterator_2->set_body(body_2);
    tensor_iterator_2->set_sliced_input(X_2, inbetween_squeeze, 0, 1, 1, -1, 0);
    tensor_iterator_2->set_merged_input(H_t_2, hidden_memory_2_constant, H_o_2);
    tensor_iterator_2->set_merged_input(C_t_2, cell_memory_2_constant, C_o_2);

    auto out_unsqueeze_2 = tensor_iterator_2->get_iter_value(unsqueeze_o_2, -1);
    auto out_hidden_2 = tensor_iterator_2->get_iter_value(H_o_2, -1);
    auto out_cell_2 = tensor_iterator_2->get_iter_value(C_o_2, -1);

    tensor_iterator_2->validate_and_infer_types();
    auto final_reshape_pattern =
        std::make_shared<ov::op::v0::Constant>(element::i64, Shape{4}, std::vector<size_t>({1, 1, 1, hiddenSize}));
    auto final_reshape = std::make_shared<ov::op::v1::Reshape>(out_unsqueeze_2, final_reshape_pattern, false);
    final_reshape->set_friendly_name("Reshape_1");

    functionRefs = std::make_shared<Model>(final_reshape, input_parameter, "PureTI");
}

void MultipleLSTMCellTest::init_memory() {
    auto states = inferRequest.query_state();
    for (auto& state : states) {
        auto name = state.get_name();
        auto tensor = state.get_state();
        if (name.find("cell_state_1") != std::string::npos || name.find("cell_state_2") != std::string::npos) {
            std::memcpy(tensor.data(), cell_memory_init.data(), cell_memory_init.size() * sizeof(float));
        } else if (name.find("hidden_state_1") != std::string::npos ||
                   name.find("hidden_state_2") != std::string::npos) {
            std::memcpy(tensor.data(), hidden_memory_init.data(), hidden_memory_init.size() * sizeof(float));
        } else {
            GTEST_FAIL() << "unknown memory state";
        }
        state.set_state(tensor);
    }
}

void MultipleLSTMCellTest::apply_low_latency() {
    // Calculate values after LowLatency transformation
    create_pure_tensor_iterator_model();
    function = functionRefs;
    if (transformation == ov::test::utils::MemoryTransformation::LOW_LATENCY_V2) {
        function->validate_nodes_and_infer_types();
        // Apply LowLatency (insert Assigns/ReadValues) and UnrollTensorIterator

        ov::pass::Manager manager;
        manager.register_pass<pass::LowLatency2>();
        manager.run_passes(function);
        bool ti_found = ov::test::utils::is_tensor_iterator_exist(function);
        EXPECT_EQ(ti_found, false);
        compile_model();
    }
}

void MultipleLSTMCellTest::compile_model() {
    ov::test::SubgraphBaseStaticTest::compile_model();
    inferRequest = compiledModel.create_infer_request();
}

void MultipleLSTMCellTest::infer() {
    for (const auto& input : inputs) {
        inferRequest.set_tensor(input.first, input.second);
    }
    inferRequest.infer();
}

void MultipleLSTMCellTest::run() {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    if (transformation != ov::test::utils::MemoryTransformation::NONE) {
        apply_low_latency();
    } else {
        compile_model();
    }

    init_memory();
    generate_inputs(input_shapes);

    // Calculate ref values
    if (transformation == ov::test::utils::MemoryTransformation::NONE) {
        switch_to_friendly_model();
    } else {
        create_pure_tensor_iterator_model();
    }
    validate();
}

}  // namespace test
}  // namespace ov
