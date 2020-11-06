// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <tuple>
#include <string>
#include <vector>
#include <memory>
#include <functional>

#include "ie_core.hpp"

#include "common_test_utils/common_utils.hpp"
#include "functional_test_utils/blob_utils.hpp"
#include "functional_test_utils/precision_utils.hpp"
#include "functional_test_utils/plugin_cache.hpp"
#include "functional_test_utils/skip_tests_config.hpp"
#include "ngraph_functions/utils/ngraph_helpers.hpp"
#include "ngraph_functions/builders.hpp"

#include <transformations/op_conversions/lstm_cell_decomposition.hpp>
#include <ngraph/pass/visualize_tree.hpp>
#include <ie_transformations.hpp>
#include "transformations/control_flow/unroll_tensor_iterator.hpp"
#include "transformations/common_optimizations/low_latency.hpp"
#include "subgraph_tests/multiple_LSTMCell.hpp"

namespace SubgraphTestsDefinitions {
std::string MultipleLSTMCellTest::getTestCaseName(const testing::TestParamInfo<multipleLSTMCellParams> &obj) {
    std::string targetDevice;
    InferenceEngine::Precision netPrecision;
    size_t inputSize;
    size_t hiddenSize;
    std::map<std::string, std::string> config;
    std::tie(targetDevice, netPrecision, inputSize, hiddenSize, config) = obj.param;
    std::ostringstream result;

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
    std::tie(targetDevice, netPrecision, inputSize, hiddenSize, config) = this->GetParam();
    configuration.insert(config.begin(), config.end());
    auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);

    std::vector<size_t> input_dims { 1, inputSize };
    std::vector<size_t> squeeze_axes {0};
    std::vector<size_t> hidden_memory_dims {1, hiddenSize};
    std::vector<size_t> cell_memory_dims {1, hiddenSize};

    const int seed = 0;
    std::mt19937 gen(static_cast<float>(seed));

    auto generateFloatNumbers = [gen](std::size_t vec_len, float min, float max) mutable {
        std::vector<float> res;

        std::uniform_real_distribution<float> dist(min, max);
        for (int i = 0; i < vec_len; i++)
            res.emplace_back(static_cast<float>(dist(gen)));

        return res;
    };

    input_bias = generateFloatNumbers(inputSize, -0.25f, 0.0f);
    input_weights = generateFloatNumbers(inputSize, 0.0f, 0.15f);
    hidden_memory_init = generateFloatNumbers(hiddenSize, -0.2f, 0.2f);
    cell_memory_init = generateFloatNumbers(hiddenSize, -0.2f, 0.2f);
    weights_vals = generateFloatNumbers(4 * hiddenSize * inputSize, -0.1f, 0.1f);
    weights_2_vals = generateFloatNumbers(4 * hiddenSize * hiddenSize, -0.1f, 0.1f);
    reccurrenceWeights_vals = generateFloatNumbers(4 * hiddenSize * hiddenSize, -0.1f, 0.1f);
    bias_vals = generateFloatNumbers(4 * hiddenSize, -0.25f, 0.15f);

    auto input_parameter = ngraph::builder::makeParams(ngPrc, {input_dims});

    auto input_add_const = ngraph::builder::makeConstant(ngPrc, input_dims, input_bias);
    auto add = ngraph::builder::makeEltwise(input_parameter[0], input_add_const, ngraph::helpers::EltwiseTypes::ADD);

    auto input_mul_const = ngraph::builder::makeConstant(ngPrc, input_dims, input_weights);
    auto mul = ngraph::builder::makeEltwise(add, input_mul_const, ngraph::helpers::EltwiseTypes::MULTIPLY);

    auto unsqueeze_input_const = std::make_shared<ngraph::op::Constant>(ngraph::element::i64, ngraph::Shape{1}, squeeze_axes);
    auto unsqueeze_input = std::make_shared<ngraph::op::Unsqueeze>(mul, unsqueeze_input_const);

    auto permute_in_params = std::make_shared<ngraph::opset1::Constant>(ngraph::element::i64, ngraph::Shape{3}, ngraph::Shape{{1, 0, 2}});
    auto permute_in = std::make_shared<ngraph::opset1::Transpose>(unsqueeze_input, permute_in_params);

    auto cell_memory_constant = ngraph::builder::makeConstant<float>(ngPrc, cell_memory_dims, cell_memory_init);
    auto cell_memory_read = std::make_shared<ngraph::op::ReadValue>(cell_memory_constant, "cell_memory");
    cell_memory_read->set_friendly_name("cell_memory");

    auto hidden_memory_constant = ngraph::builder::makeConstant<float>(ngPrc, hidden_memory_dims, hidden_memory_init);
    auto hidden_memory_read = std::make_shared<ngraph::op::ReadValue>(hidden_memory_constant, "hidden_memory");
    hidden_memory_read->set_friendly_name("hidden_memory");

    // Body - inputs
    auto X = std::make_shared<ngraph::op::Parameter>(ngPrc, ngraph::Shape{1, 1, inputSize});
    auto H_t = std::make_shared<ngraph::op::Parameter>(ngPrc, ngraph::Shape{1, hiddenSize});
    auto C_t = std::make_shared<ngraph::op::Parameter>(ngPrc, ngraph::Shape{1, hiddenSize});
    // Body - layers
    auto squeeze_const = std::make_shared<ngraph::op::Constant>(ngraph::element::i64, ngraph::Shape{1}, squeeze_axes);
    auto squeeze = std::make_shared<ngraph::op::Squeeze>(X, squeeze_const);

    auto weightsNode = ngraph::builder::makeConstant<float>(ngPrc, { 4 * hiddenSize, inputSize }, weights_vals);
    auto reccurrenceWeightsNode = ngraph::builder::makeConstant<float>(ngPrc, { 4 * hiddenSize, hiddenSize }, reccurrenceWeights_vals);
    auto biasNode = ngraph::builder::makeConstant<float>(ngPrc, {4 * hiddenSize}, bias_vals);
    auto lstm = std::make_shared<ngraph::opset4::LSTMCell>(squeeze, H_t, C_t, weightsNode, reccurrenceWeightsNode, biasNode, hiddenSize);

    auto unsqueeze_const = std::make_shared<ngraph::op::Constant>(ngraph::element::i64, ngraph::Shape{1}, squeeze_axes);
    auto unsqueeze = std::make_shared<ngraph::op::Unsqueeze>(lstm->output(0), unsqueeze_const);
    // body - outputs
    auto H_o = lstm->output(0);
    auto C_o = lstm->output(1);
    auto unsqueeze_o = unsqueeze->output(0);

    auto body = std::make_shared<ngraph::Function>(ngraph::OutputVector{unsqueeze_o, H_o, C_o}, ngraph::ParameterVector {X, H_t, C_t});
    // TI construction
    auto tensor_iterator = std::make_shared<ngraph::op::TensorIterator>();
    tensor_iterator->set_body(body);
    tensor_iterator->set_invariant_input(X, permute_in);
    tensor_iterator->set_merged_input(H_t, hidden_memory_read, H_o);
    tensor_iterator->set_merged_input(C_t, cell_memory_read, C_o);

    auto out_unsqueeze = tensor_iterator->get_iter_value(unsqueeze_o, -1);
    auto out_hidden = tensor_iterator->get_iter_value(H_o, -1);
    auto out_cell = tensor_iterator->get_iter_value(C_o, -1);

    out_hidden.get_tensor().set_element_type(ngPrc);
    out_cell.get_tensor().set_element_type(ngPrc);

    auto cell_memory_write = std::make_shared<ngraph::op::Assign>(out_cell, "cell_memory");
    auto hidden_memory_write = std::make_shared<ngraph::op::Assign>(out_hidden, "hidden_memory");

    auto first_reshape_pattern = std::make_shared<ngraph::op::Constant>(ngraph::element::i64,
                                                                        ngraph::Shape{4}, std::vector<size_t>({1, 1, 1, hiddenSize}));
    auto first_reshape = std::make_shared<ngraph::op::v1::Reshape>(out_unsqueeze, first_reshape_pattern, false);
    // End of TI 1

    auto inbetween_squeeze_const = std::make_shared<ngraph::op::Constant>(ngraph::element::i64, ngraph::Shape{1}, squeeze_axes);
    auto inbetween_squeeze = std::make_shared<ngraph::op::Squeeze>(first_reshape, inbetween_squeeze_const);

    // Second TI
    auto cell_memory_2_constant = ngraph::builder::makeConstant<float>(ngPrc, cell_memory_dims, cell_memory_init);
    auto cell_memory_2_read = std::make_shared<ngraph::op::ReadValue>(cell_memory_2_constant, "cell_memory_2");
    cell_memory_2_read->set_friendly_name("cell_memory_2");

    auto hidden_memory_2_constant = ngraph::builder::makeConstant<float>(ngPrc, hidden_memory_dims, hidden_memory_init);
    auto hidden_memory_2_read = std::make_shared<ngraph::op::ReadValue>(hidden_memory_2_constant, "hidden_memory_2");
    hidden_memory_2_read->set_friendly_name("hidden_memory_2");

    // Body - inputs
    auto X_2 = std::make_shared<ngraph::op::Parameter>(ngPrc, ngraph::Shape{1, 1, hiddenSize});
    auto H_t_2 = std::make_shared<ngraph::op::Parameter>(ngPrc, ngraph::Shape{1, hiddenSize});
    auto C_t_2 = std::make_shared<ngraph::op::Parameter>(ngPrc, ngraph::Shape{1, hiddenSize});
    // Body - layers
    auto squeeze_2_const = std::make_shared<ngraph::op::Constant>(ngraph::element::i64, ngraph::Shape{1}, squeeze_axes);
    auto squeeze_2 = std::make_shared<ngraph::op::Squeeze>(X_2, squeeze_2_const);

    auto weightsNode_2 = ngraph::builder::makeConstant<float>(ngPrc, { 4 * hiddenSize, hiddenSize }, weights_2_vals);
    auto reccurrenceWeightsNode_2 = ngraph::builder::makeConstant<float>(ngPrc, { 4 * hiddenSize, hiddenSize }, reccurrenceWeights_vals);
    auto biasNode_2 = ngraph::builder::makeConstant<float>(ngPrc, {4 * hiddenSize}, bias_vals);
    auto lstm_2 = std::make_shared<ngraph::opset4::LSTMCell>(squeeze_2, H_t_2, C_t_2, weightsNode_2, reccurrenceWeightsNode_2, biasNode_2, hiddenSize);

    auto unsqueeze_2_const = std::make_shared<ngraph::op::Constant>(ngraph::element::i64, ngraph::Shape{1}, squeeze_axes);
    auto unsqueeze_2 = std::make_shared<ngraph::op::Unsqueeze>(lstm_2->output(0), unsqueeze_2_const);
    // body - outputs
    auto H_o_2 = lstm_2->output(0);
    auto C_o_2 = lstm_2->output(1);
    auto unsqueeze_o_2 = unsqueeze_2->output(0);

    auto body_2 = std::make_shared<ngraph::Function>(ngraph::OutputVector{unsqueeze_o_2, H_o_2, C_o_2}, ngraph::ParameterVector {X_2, H_t_2, C_t_2});
    // TI construction
    auto tensor_iterator_2 = std::make_shared<ngraph::op::TensorIterator>();
    tensor_iterator_2->set_body(body_2);
    tensor_iterator_2->set_invariant_input(X_2, inbetween_squeeze);
    tensor_iterator_2->set_merged_input(H_t_2, hidden_memory_2_read, H_o_2);
    tensor_iterator_2->set_merged_input(C_t_2, cell_memory_2_read, C_o_2);

    auto out_unsqueeze_2 = tensor_iterator_2->get_iter_value(unsqueeze_o_2, -1);
    auto out_hidden_2 = tensor_iterator_2->get_iter_value(H_o_2, -1);
    auto out_cell_2 = tensor_iterator_2->get_iter_value(C_o_2, -1);

    out_hidden_2.get_tensor().set_element_type(ngPrc);
    out_cell_2.get_tensor().set_element_type(ngPrc);

    auto cell_memory_2_write = std::make_shared<ngraph::op::Assign>(out_cell_2, "cell_memory_2");
    auto hidden_memory_2_write = std::make_shared<ngraph::op::Assign>(out_hidden_2, "hidden_memory_2");

    auto final_reshape_pattern = std::make_shared<ngraph::op::Constant>(ngraph::element::i64,
                                                                        ngraph::Shape{4}, std::vector<size_t>({1, 1, 1, hiddenSize}));
    auto final_reshape = std::make_shared<ngraph::op::v1::Reshape>(out_unsqueeze_2, final_reshape_pattern, false);

    cell_memory_write->add_control_dependency(cell_memory_read);
    final_reshape->add_control_dependency(cell_memory_write);

    hidden_memory_write->add_control_dependency(hidden_memory_read);
    final_reshape->add_control_dependency(hidden_memory_write);

    cell_memory_2_write->add_control_dependency(cell_memory_2_read);
    final_reshape->add_control_dependency(cell_memory_2_write);

    hidden_memory_2_write->add_control_dependency(hidden_memory_2_read);
    final_reshape->add_control_dependency(hidden_memory_2_write);

    function = std::make_shared<ngraph::Function>(final_reshape, input_parameter, "TI_with_memory");
}

void MultipleLSTMCellTest::switchToNgraphFriendlyModel() {
    InferenceEngine::Precision netPrecision;
    std::map<std::string, std::string> config;
    size_t inputSize;
    std::tie(targetDevice, netPrecision, inputSize, hiddenSize, config) = this->GetParam();
    auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);

    std::vector<size_t> input_dims { 1, inputSize };
    std::vector<size_t> squeeze_axes {0};
    std::vector<size_t> hidden_memory_dims {1, hiddenSize};
    std::vector<size_t> cell_memory_dims {1, hiddenSize};

    auto input_parameter = ngraph::builder::makeParams(ngPrc, {input_dims});

    auto input_add_const = ngraph::builder::makeConstant(ngPrc, input_dims, input_bias);
    auto add = ngraph::builder::makeEltwise(input_parameter[0], input_add_const, ngraph::helpers::EltwiseTypes::ADD);

    auto input_mul_const = ngraph::builder::makeConstant(ngPrc, input_dims, input_weights);
    auto mul = ngraph::builder::makeEltwise(add, input_mul_const, ngraph::helpers::EltwiseTypes::MULTIPLY);

    auto unsqueeze_input_const = std::make_shared<ngraph::op::Constant>(ngraph::element::i64, ngraph::Shape{1}, squeeze_axes);
    auto unsqueeze_input = std::make_shared<ngraph::op::Unsqueeze>(mul, unsqueeze_input_const);

    // Body 1 - layers
    auto cell_memory_constant = ngraph::builder::makeConstant<float>(ngPrc, cell_memory_dims, cell_memory_init);

    auto hidden_memory_constant = ngraph::builder::makeConstant<float>(ngPrc, hidden_memory_dims, hidden_memory_init);

    auto squeeze_const = std::make_shared<ngraph::op::Constant>(ngraph::element::i64, ngraph::Shape{1}, squeeze_axes);
    auto squeeze = std::make_shared<ngraph::op::Squeeze>(unsqueeze_input, squeeze_const);

    auto weightsNode = ngraph::builder::makeConstant<float>(ngPrc, { 4 * hiddenSize, inputSize }, weights_vals);
    auto reccurrenceWeightsNode = ngraph::builder::makeConstant<float>(ngPrc, { 4 * hiddenSize, hiddenSize }, reccurrenceWeights_vals);
    auto biasNode = ngraph::builder::makeConstant<float>(ngPrc, {4 * hiddenSize}, bias_vals);
    auto lstm = std::make_shared<ngraph::opset4::LSTMCell>(squeeze, hidden_memory_constant, cell_memory_constant, weightsNode,
                                                           reccurrenceWeightsNode, biasNode, hiddenSize);

    auto unsqueeze_const = std::make_shared<ngraph::op::Constant>(ngraph::element::i64, ngraph::Shape{1}, squeeze_axes);
    auto unsqueeze = std::make_shared<ngraph::op::Unsqueeze>(lstm->output(0), unsqueeze_const);

    auto first_reshape_pattern = std::make_shared<ngraph::op::Constant>(ngraph::element::i64,
                                                                        ngraph::Shape{4}, std::vector<size_t>({1, 1, 1, hiddenSize}));
    auto first_reshape = std::make_shared<ngraph::op::v1::Reshape>(unsqueeze, first_reshape_pattern, false);
    // Body 1 - end

    auto inbetween_squeeze_const = std::make_shared<ngraph::op::Constant>(ngraph::element::i64, ngraph::Shape{1}, squeeze_axes);
    auto inbetween_squeeze = std::make_shared<ngraph::op::Squeeze>(first_reshape, inbetween_squeeze_const);

    // Body 2 - layers
    auto cell_memory_2_constant = ngraph::builder::makeConstant<float>(ngPrc, cell_memory_dims, cell_memory_init);

    auto hidden_memory_2_constant = ngraph::builder::makeConstant<float>(ngPrc, hidden_memory_dims, hidden_memory_init);

    auto squeeze_2_const = std::make_shared<ngraph::op::Constant>(ngraph::element::i64, ngraph::Shape{1}, squeeze_axes);
    auto squeeze_2 = std::make_shared<ngraph::op::Squeeze>(inbetween_squeeze, squeeze_2_const);

    auto weightsNode_2 = ngraph::builder::makeConstant<float>(ngPrc, { 4 * hiddenSize, hiddenSize }, weights_2_vals);
    auto reccurrenceWeightsNode_2 = ngraph::builder::makeConstant<float>(ngPrc, { 4 * hiddenSize, hiddenSize }, reccurrenceWeights_vals);
    auto biasNode_2 = ngraph::builder::makeConstant<float>(ngPrc, {4 * hiddenSize}, bias_vals);
    auto lstm_2 = std::make_shared<ngraph::opset4::LSTMCell>(squeeze_2, hidden_memory_2_constant, cell_memory_2_constant, weightsNode_2,
        reccurrenceWeightsNode_2, biasNode_2, hiddenSize);

    auto unsqueeze_2_const = std::make_shared<ngraph::op::Constant>(ngraph::element::i64, ngraph::Shape{1}, squeeze_axes);
    auto unsqueeze_2 = std::make_shared<ngraph::op::Unsqueeze>(lstm_2->output(0), unsqueeze_2_const);

    auto final_reshape_pattern = std::make_shared<ngraph::op::Constant>(ngraph::element::i64,
        ngraph::Shape{4}, std::vector<size_t>({1, 1, 1, hiddenSize}));
    auto final_reshape = std::make_shared<ngraph::op::v1::Reshape>(unsqueeze_2, final_reshape_pattern, false);
    // Body 2 - end

    function = std::make_shared<ngraph::Function>(final_reshape, input_parameter, "TI_unrolled_without_memory");
}

void MultipleLSTMCellTest::CreatePureTensorIteratorModel() {
    InferenceEngine::Precision netPrecision;
    std::map<std::string, std::string> config;
    size_t inputSize;
    std::tie(targetDevice, netPrecision, inputSize, hiddenSize, config) = this->GetParam();
    auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);

    std::vector<size_t> input_dims { 1, inputSize };
    std::vector<size_t> squeeze_axes {0};
    std::vector<size_t> hidden_memory_dims {1, hiddenSize};
    std::vector<size_t> cell_memory_dims {1, hiddenSize};

    auto input_parameter = ngraph::builder::makeParams(ngPrc, {input_dims});

    auto input_add_const = ngraph::builder::makeConstant(ngPrc, input_dims, input_bias);
    auto add = ngraph::builder::makeEltwise(input_parameter[0], input_add_const, ngraph::helpers::EltwiseTypes::ADD);

    auto input_mul_const = ngraph::builder::makeConstant(ngPrc, input_dims, input_weights);
    auto mul = ngraph::builder::makeEltwise(add, input_mul_const, ngraph::helpers::EltwiseTypes::MULTIPLY);

    auto unsqueeze_input_const = std::make_shared<ngraph::op::Constant>(ngraph::element::i64, ngraph::Shape{1}, squeeze_axes);
    auto unsqueeze_input = std::make_shared<ngraph::op::Unsqueeze>(mul, unsqueeze_input_const);

    auto permute_in_params = std::make_shared<ngraph::opset1::Constant>(ngraph::element::i64, ngraph::Shape{3}, ngraph::Shape{{1, 0, 2}});
    auto permute_in = std::make_shared<ngraph::opset1::Transpose>(unsqueeze_input, permute_in_params);

    auto cell_memory_constant = ngraph::builder::makeConstant<float>(ngPrc, cell_memory_dims, cell_memory_init);

    auto hidden_memory_constant = ngraph::builder::makeConstant<float>(ngPrc, hidden_memory_dims, hidden_memory_init);

    // Body - inputs
    auto X = std::make_shared<ngraph::op::Parameter>(ngPrc, ngraph::Shape{1, 1, inputSize});
    auto H_t = std::make_shared<ngraph::op::Parameter>(ngPrc, ngraph::Shape{1, hiddenSize});
    auto C_t = std::make_shared<ngraph::op::Parameter>(ngPrc, ngraph::Shape{1, hiddenSize});
    H_t->set_friendly_name("hidden_state_1");
    C_t->set_friendly_name("cell_state_1");
    // Body - layers
    auto squeeze_const = std::make_shared<ngraph::op::Constant>(ngraph::element::i64, ngraph::Shape{1}, squeeze_axes);
    auto squeeze = std::make_shared<ngraph::op::Squeeze>(X, squeeze_const);

    auto weightsNode = ngraph::builder::makeConstant<float>(ngPrc, { 4 * hiddenSize, inputSize }, weights_vals);
    auto reccurrenceWeightsNode = ngraph::builder::makeConstant<float>(ngPrc, { 4 * hiddenSize, hiddenSize }, reccurrenceWeights_vals);
    auto biasNode = ngraph::builder::makeConstant<float>(ngPrc, {4 * hiddenSize}, bias_vals);
    auto lstm = std::make_shared<ngraph::opset4::LSTMCell>(squeeze, H_t, C_t, weightsNode, reccurrenceWeightsNode, biasNode, hiddenSize);

    auto unsqueeze_const = std::make_shared<ngraph::op::Constant>(ngraph::element::i64, ngraph::Shape{1}, squeeze_axes);
    auto unsqueeze = std::make_shared<ngraph::op::Unsqueeze>(lstm->output(0), unsqueeze_const);
    // body - outputs
    auto H_o = lstm->output(0);
    auto C_o = lstm->output(1);
    auto unsqueeze_o = unsqueeze->output(0);

    auto body = std::make_shared<ngraph::Function>(ngraph::OutputVector{unsqueeze_o, H_o, C_o}, ngraph::ParameterVector {X, H_t, C_t});
    // TI construction
    auto tensor_iterator = std::make_shared<ngraph::op::TensorIterator>();
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

    auto first_reshape_pattern = std::make_shared<ngraph::op::Constant>(ngraph::element::i64,
                                                                        ngraph::Shape{4}, std::vector<size_t>({1, 1, 1, hiddenSize}));
    auto first_reshape = std::make_shared<ngraph::op::v1::Reshape>(out_unsqueeze, first_reshape_pattern, false);
    // End of TI 1

    auto inbetween_squeeze_const = std::make_shared<ngraph::op::Constant>(ngraph::element::i64, ngraph::Shape{1}, squeeze_axes);
    auto inbetween_squeeze = std::make_shared<ngraph::op::Squeeze>(first_reshape, inbetween_squeeze_const);

    // Second TI
    auto cell_memory_2_constant = ngraph::builder::makeConstant<float>(ngPrc, cell_memory_dims, cell_memory_init);

    auto hidden_memory_2_constant = ngraph::builder::makeConstant<float>(ngPrc, hidden_memory_dims, hidden_memory_init);

    // Body - inputs
    auto X_2 = std::make_shared<ngraph::op::Parameter>(ngPrc, ngraph::Shape{1, 1, hiddenSize});
    auto H_t_2 = std::make_shared<ngraph::op::Parameter>(ngPrc, ngraph::Shape{1, hiddenSize});
    auto C_t_2 = std::make_shared<ngraph::op::Parameter>(ngPrc, ngraph::Shape{1, hiddenSize});
    H_t_2->set_friendly_name("hidden_state_2");
    C_t_2->set_friendly_name("cell_state_2");
    // Body - layers
    auto squeeze_2_const = std::make_shared<ngraph::op::Constant>(ngraph::element::i64, ngraph::Shape{1}, squeeze_axes);
    auto squeeze_2 = std::make_shared<ngraph::op::Squeeze>(X_2, squeeze_2_const);

    auto weightsNode_2 = ngraph::builder::makeConstant<float>(ngPrc, { 4 * hiddenSize, hiddenSize }, weights_2_vals);
    auto reccurrenceWeightsNode_2 = ngraph::builder::makeConstant<float>(ngPrc, { 4 * hiddenSize, hiddenSize }, reccurrenceWeights_vals);
    auto biasNode_2 = ngraph::builder::makeConstant<float>(ngPrc, {4 * hiddenSize}, bias_vals);
    auto lstm_2 = std::make_shared<ngraph::opset4::LSTMCell>(squeeze_2, H_t_2, C_t_2, weightsNode_2, reccurrenceWeightsNode_2, biasNode_2, hiddenSize);

    auto unsqueeze_2_const = std::make_shared<ngraph::op::Constant>(ngraph::element::i64, ngraph::Shape{1}, squeeze_axes);
    auto unsqueeze_2 = std::make_shared<ngraph::op::Unsqueeze>(lstm_2->output(0), unsqueeze_2_const);
    // body - outputs
    auto H_o_2 = lstm_2->output(0);
    auto C_o_2 = lstm_2->output(1);
    auto unsqueeze_o_2 = unsqueeze_2->output(0);

    auto body_2 = std::make_shared<ngraph::Function>(ngraph::OutputVector{unsqueeze_o_2, H_o_2, C_o_2}, ngraph::ParameterVector {X_2, H_t_2, C_t_2});
    // TI construction
    auto tensor_iterator_2 = std::make_shared<ngraph::op::TensorIterator>();
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
    auto final_reshape_pattern = std::make_shared<ngraph::op::Constant>(ngraph::element::i64,
                                                                        ngraph::Shape{4}, std::vector<size_t>({1, 1, 1, hiddenSize}));
    auto final_reshape = std::make_shared<ngraph::op::v1::Reshape>(out_unsqueeze_2, final_reshape_pattern, false);

    function = std::make_shared<ngraph::Function>(final_reshape, input_parameter, "PureTI");
}

void MultipleLSTMCellTest::Run() {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    InferenceEngine::TensorDesc state_description(InferenceEngine::Precision::FP32,
                                                  InferenceEngine::SizeVector({1, hiddenSize}),
                                                  InferenceEngine::Layout::NC);
    LoadNetwork();
    auto states = executableNetwork.QueryState();
    for (auto& state : states) {
        auto name = state.GetName();
        if (name == "cell_memory") {
            auto blob = FuncTestUtils::createAndFillBlobWithFloatArray(state_description,
                                                                       cell_memory_init.data(), cell_memory_init.size());
            state.SetState(blob);
        } else if (name == "hidden_memory") {
            auto blob = FuncTestUtils::createAndFillBlobWithFloatArray(state_description,
                                                                       hidden_memory_init.data(), hidden_memory_init.size());
            state.SetState(blob);
        } else if (name == "cell_memory_2") {
            auto blob = FuncTestUtils::createAndFillBlobWithFloatArray(state_description,
                cell_memory_init.data(), cell_memory_init.size());
            state.SetState(blob);
        } else if (name == "hidden_memory_2") {
            auto blob = FuncTestUtils::createAndFillBlobWithFloatArray(state_description,
                hidden_memory_init.data(), hidden_memory_init.size());
            state.SetState(blob);
        } else {
            GTEST_FAIL() << "unknown memory state";
        }
    }
    Infer();
    switchToNgraphFriendlyModel();
    Validate();
}

void MultipleLSTMCellTest::RunLowLatency(bool regular_api) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    InferenceEngine::TensorDesc state_description(InferenceEngine::Precision::FP32,
                                                  InferenceEngine::SizeVector({1, hiddenSize}),
                                                  InferenceEngine::Layout::NC);
    // Calculate values after LowLatency transformation
    CreatePureTensorIteratorModel();
    if (regular_api) {
        cnnNetwork = InferenceEngine::CNNNetwork{function};
        InferenceEngine::LowLatency(cnnNetwork);
        ConfigureNetwork();
        executableNetwork = core->LoadNetwork(cnnNetwork, targetDevice, configuration);
    } else {
        function->validate_nodes_and_infer_types();
        // Apply LowLatency (insert Assigns/ReadValues) and UnrollTensorIterator
        ngraph::pass::Manager manager;
        manager.register_pass<ngraph::pass::LowLatency>(); // LowLatency enables UnrollTI
        manager.run_passes(function);
        LoadNetwork();
    }
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
    Infer();

    // Calculate ref values for Unrolled TI
    CreatePureTensorIteratorModel();
    ngraph::pass::Manager manager_2;
    manager_2.register_pass<ngraph::pass::UnrollTensorIterator>();
    manager_2.run_passes(function);
    Validate();
}

TEST_P(MultipleLSTMCellTest, CompareWithRefs) {
    Run();
};

TEST_P(MultipleLSTMCellTest, CompareWithRefs_LowLatencyTransformation) {
    RunLowLatency();
};

TEST_P(MultipleLSTMCellTest, CompareWithRefs_LowLatencyRegularAPITransformation) {
    RunLowLatency(true);
};
}  // namespace SubgraphTestsDefinitions
