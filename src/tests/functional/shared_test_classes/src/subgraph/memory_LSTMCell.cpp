// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <ie_transformations.hpp>
#include <transformations/control_flow/unroll_tensor_iterator.hpp>

#include "ngraph/pass/low_latency.hpp"
#include "ov_models/builders.hpp"
#include "shared_test_classes/subgraph/memory_LSTMCell.hpp"
#include "functional_test_utils/core_config.hpp"

using namespace ngraph;
using namespace opset7;

namespace SubgraphTestsDefinitions {

    std::string MemoryLSTMCellTest::getTestCaseName(const testing::TestParamInfo<memoryLSTMCellParams> &obj) {
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

    size_t hiddenSize;


    void MemoryLSTMCellTest::SetUp() {
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

        input_bias = ov::test::utils::generate_float_numbers(inputSize, -0.2f, 0.0f);
        input_weights = ov::test::utils::generate_float_numbers(inputSize, 0.0f, 0.1f);
        hidden_memory_init = ov::test::utils::generate_float_numbers(hiddenSize, -0.2f, 0.2f);
        cell_memory_init = ov::test::utils::generate_float_numbers(hiddenSize, -0.2f, 0.2f);
        weights_vals = ov::test::utils::generate_float_numbers(4 * hiddenSize * inputSize, -0.1f, 0.1f);
        reccurrenceWeights_vals = ov::test::utils::generate_float_numbers(4 * hiddenSize * hiddenSize, -0.1f, 0.1f);
        bias_vals = ov::test::utils::generate_float_numbers(4 * hiddenSize, -0.2f, 0.1f);

        ov::ParameterVector input_parameter {std::make_shared<ov::op::v0::Parameter>(ngPrc, ov::Shape(input_dims))};

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
                std::make_shared<Variable>(VariableInfo{PartialShape(cell_memory_dims), ngPrc, "cell_state_1"});
        auto var_hidden =
                std::make_shared<Variable>(VariableInfo{PartialShape(cell_memory_dims), ngPrc, "hidden_state_1"});
        auto cell_memory_read = std::make_shared<ReadValue>(cell_memory_constant, var_cell);

        auto hidden_memory_constant = builder::makeConstant<float>(ngPrc, hidden_memory_dims, hidden_memory_init);
        auto hidden_memory_read = std::make_shared<ReadValue>(hidden_memory_constant, var_hidden);

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
        tensor_iterator->set_sliced_input(X, permute_in, 0, 1, 1, -1, 0);
        tensor_iterator->set_merged_input(H_t, hidden_memory_read, H_o);
        tensor_iterator->set_merged_input(C_t, cell_memory_read, C_o);

        auto out_unsqueeze = tensor_iterator->get_iter_value(unsqueeze_o, -1);
        auto out_hidden = tensor_iterator->get_iter_value(H_o, -1);
        auto out_cell = tensor_iterator->get_iter_value(C_o, -1);


        out_hidden.get_tensor().set_element_type(ngPrc);
        out_cell.get_tensor().set_element_type(ngPrc);

        auto cell_memory_write = std::make_shared<Assign>(out_cell, var_cell);
        auto hidden_memory_write = std::make_shared<Assign>(out_hidden, var_hidden);

        auto final_reshape_pattern = std::make_shared<Constant>(element::i64, Shape{4},
                                                                                std::vector<size_t>({1, 1, 1, hiddenSize}));
        auto final_reshape = std::make_shared<Reshape>(out_unsqueeze, final_reshape_pattern, false);

        cell_memory_write->add_control_dependency(cell_memory_read);
        hidden_memory_write->add_control_dependency(hidden_memory_read);

        function = std::make_shared<Function>(OutputVector{final_reshape},
                                              SinkVector{cell_memory_write, hidden_memory_write},
                                              input_parameter,
                                              "TI_with_memory");
        tensor_iterator->validate_and_infer_types();
    }

    void MemoryLSTMCellTest::switchToNgraphFriendlyModel() {
        InferenceEngine::Precision netPrecision;
        std::map<std::string, std::string> config;
        size_t inputSize;
        std::tie(transformation, targetDevice, netPrecision, inputSize, hiddenSize, config) = this->GetParam();
        auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);

        std::vector<size_t> input_dims { 1, inputSize };
        std::vector<size_t> squeeze_axes {0};
        std::vector<size_t> hidden_memory_dims {1, hiddenSize};
        std::vector<size_t> cell_memory_dims {1, hiddenSize};

        ov::ParameterVector input_parameter {std::make_shared<ov::op::v0::Parameter>(ngPrc, ov::Shape(input_dims))};

        auto input_add_const = builder::makeConstant(ngPrc, input_dims, input_bias);
        auto add = builder::makeEltwise(input_parameter[0], input_add_const, helpers::EltwiseTypes::ADD);

        auto input_mul_const = builder::makeConstant(ngPrc, input_dims, input_weights);
        auto mul = builder::makeEltwise(add, input_mul_const, helpers::EltwiseTypes::MULTIPLY);

        auto unsqueeze_input_const = std::make_shared<Constant>(element::i64, Shape{1}, squeeze_axes);
        auto unsqueeze_input = std::make_shared<Unsqueeze>(mul, unsqueeze_input_const);

        auto cell_memory_constant = builder::makeConstant<float>(ngPrc, cell_memory_dims, cell_memory_init);

        auto hidden_memory_constant = builder::makeConstant<float>(ngPrc, hidden_memory_dims, hidden_memory_init);

        // Body - layers
        auto squeeze_const = std::make_shared<Constant>(element::i64, Shape{1}, squeeze_axes);
        auto squeeze = std::make_shared<Squeeze>(unsqueeze_input, squeeze_const);

        auto weightsNode = builder::makeConstant<float>(ngPrc, { 4 * hiddenSize, inputSize }, weights_vals);
        auto reccurrenceWeightsNode = builder::makeConstant<float>(ngPrc, { 4 * hiddenSize, hiddenSize }, reccurrenceWeights_vals);
        auto biasNode = builder::makeConstant<float>(ngPrc, {4 * hiddenSize}, bias_vals);
        auto lstm = std::make_shared<LSTMCell>(squeeze, hidden_memory_constant, cell_memory_constant, weightsNode,
                                                               reccurrenceWeightsNode, biasNode, hiddenSize);

        auto unsqueeze_const = std::make_shared<Constant>(element::i64, Shape{1}, squeeze_axes);
        auto unsqueeze = std::make_shared<Unsqueeze>(lstm->output(0), unsqueeze_const);

        auto final_reshape_pattern = std::make_shared<Constant>(element::i64,
                                                                            Shape{4}, std::vector<size_t>({1, 1, 1, hiddenSize}));
        auto final_reshape = std::make_shared<Reshape>(unsqueeze, final_reshape_pattern, false);

        function = std::make_shared<Function>(final_reshape, input_parameter, "TI_unrolled_without_memory");
    }

    void MemoryLSTMCellTest::CreatePureTensorIteratorModel() {
        InferenceEngine::Precision netPrecision;
        std::map<std::string, std::string> config;
        size_t inputSize;
        std::tie(transformation, targetDevice, netPrecision, inputSize, hiddenSize, config) = this->GetParam();
        auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);

        std::vector<size_t> input_dims { 1, inputSize };
        std::vector<size_t> squeeze_axes {0};
        std::vector<size_t> hidden_memory_dims {1, hiddenSize};
        std::vector<size_t> cell_memory_dims {1, hiddenSize};

        ov::ParameterVector input_parameter {std::make_shared<ov::op::v0::Parameter>(ngPrc, ov::Shape(input_dims))};

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

        auto final_reshape_pattern = std::make_shared<Constant>(element::i64, Shape{4},
                                                                                std::vector<size_t>({1, 1, 1, hiddenSize}));
        auto final_reshape = std::make_shared<Reshape>(out_unsqueeze, final_reshape_pattern, false);

        function = std::make_shared<Function>(OutputVector{final_reshape, out_hidden, out_cell}, input_parameter, "PureTI");
    }

    void MemoryLSTMCellTest::LoadNetwork() {
        LayerTestsUtils::LayerTestsCommon::LoadNetwork();
        inferRequest = executableNetwork.CreateInferRequest();
    }

    void MemoryLSTMCellTest::Infer() {
        ConfigureInferRequest();
        inferRequest.Infer();
    }

    void MemoryLSTMCellTest::Run() {
        SKIP_IF_CURRENT_TEST_IS_DISABLED()
        if (transformation != ngraph::helpers::MemoryTransformation::NONE) {
            CoreConfiguration(this);
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

    void MemoryLSTMCellTest::InitMemory() {
        auto states = inferRequest.QueryState();
        for (auto& state : states) {
            auto name = state.GetName();
            if (name.find("cell_state_1") != std::string::npos) {
                auto blob = FuncTestUtils::createAndFillBlobWithFloatArray(state.GetState()->getTensorDesc(),
                                                                           cell_memory_init.data(), cell_memory_init.size());
                state.SetState(blob);
            } else if (name.find("hidden_state_1") != std::string::npos) {
                auto blob = FuncTestUtils::createAndFillBlobWithFloatArray(state.GetState()->getTensorDesc(),
                                                                           hidden_memory_init.data(), hidden_memory_init.size());
                state.SetState(blob);
            } else {
                GTEST_FAIL() << "unknown memory state";
            }
        }
    }

    void MemoryLSTMCellTest::ApplyLowLatency() {
        // Calculate values after LowLatency transformation
        CreatePureTensorIteratorModel();
        if (transformation == ngraph::helpers::MemoryTransformation::LOW_LATENCY_V2) {
            function->validate_nodes_and_infer_types();
            // Apply LowLatency (insert Assigns/ReadValues) and UnrollTensorIterator

            pass::Manager manager;
            manager.register_pass<pass::LowLatency2>();
            manager.run_passes(function);
            bool ti_found = helpers::is_tensor_iterator_exist(function);
            EXPECT_EQ(ti_found, false);
            LoadNetwork();
        } else if (transformation == ngraph::helpers::MemoryTransformation::LOW_LATENCY_V2_REGULAR_API) {
            cnnNetwork = InferenceEngine::CNNNetwork{function};
            InferenceEngine::lowLatency2(cnnNetwork);

            bool ti_found = helpers::is_tensor_iterator_exist(cnnNetwork.getFunction());
            EXPECT_EQ(ti_found, false);

            ConfigureNetwork();
            executableNetwork = core->LoadNetwork(cnnNetwork, targetDevice, configuration);
            inferRequest = executableNetwork.CreateInferRequest();
        }
    }
}  // namespace SubgraphTestsDefinitions
