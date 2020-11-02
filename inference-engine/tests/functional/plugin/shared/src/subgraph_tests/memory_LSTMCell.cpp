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
#include "subgraph_tests/memory_LSTMCell.hpp"

namespace SubgraphTestsDefinitions {

    std::string MemoryLSTMCellTest::getTestCaseName(const testing::TestParamInfo<memoryLSTMCellParams> &obj) {
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

    size_t hiddenSize;


    void MemoryLSTMCellTest::SetUp() {
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

        auto hidden_memory_constant = ngraph::builder::makeConstant<float>(ngPrc, hidden_memory_dims, hidden_memory_init);
        auto hidden_memory_read = std::make_shared<ngraph::op::ReadValue>(hidden_memory_constant, "hidden_memory");

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

        auto final_reshape_pattern = std::make_shared<ngraph::op::Constant>(ngraph::element::i64, ngraph::Shape{4}, std::vector<size_t>({1, 1, 1, hiddenSize}));
        auto final_reshape = std::make_shared<ngraph::op::v1::Reshape>(out_unsqueeze, final_reshape_pattern, false);

        cell_memory_write->add_control_dependency(cell_memory_read);
        final_reshape->add_control_dependency(cell_memory_write);

        hidden_memory_write->add_control_dependency(hidden_memory_read);
        final_reshape->add_control_dependency(hidden_memory_write);

        function = std::make_shared<ngraph::Function>(final_reshape, input_parameter, "TI_with_memory");
    }

    void MemoryLSTMCellTest::switchToNgraphFriendlyModel() {
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

        auto cell_memory_constant = ngraph::builder::makeConstant<float>(ngPrc, cell_memory_dims, cell_memory_init);

        auto hidden_memory_constant = ngraph::builder::makeConstant<float>(ngPrc, hidden_memory_dims, hidden_memory_init);

        // Body - layers
        auto squeeze_const = std::make_shared<ngraph::op::Constant>(ngraph::element::i64, ngraph::Shape{1}, squeeze_axes);
        auto squeeze = std::make_shared<ngraph::op::Squeeze>(unsqueeze_input, squeeze_const);

        auto weightsNode = ngraph::builder::makeConstant<float>(ngPrc, { 4 * hiddenSize, inputSize }, weights_vals);
        auto reccurrenceWeightsNode = ngraph::builder::makeConstant<float>(ngPrc, { 4 * hiddenSize, hiddenSize }, reccurrenceWeights_vals);
        auto biasNode = ngraph::builder::makeConstant<float>(ngPrc, {4 * hiddenSize}, bias_vals);
        auto lstm = std::make_shared<ngraph::opset4::LSTMCell>(squeeze, hidden_memory_constant, cell_memory_constant, weightsNode,
                                                               reccurrenceWeightsNode, biasNode, hiddenSize);

        auto unsqueeze_const = std::make_shared<ngraph::op::Constant>(ngraph::element::i64, ngraph::Shape{1}, squeeze_axes);
        auto unsqueeze = std::make_shared<ngraph::op::Unsqueeze>(lstm->output(0), unsqueeze_const);

        auto final_reshape_pattern = std::make_shared<ngraph::op::Constant>(ngraph::element::i64,
                                                                            ngraph::Shape{4}, std::vector<size_t>({1, 1, 1, hiddenSize}));
        auto final_reshape = std::make_shared<ngraph::op::v1::Reshape>(unsqueeze, final_reshape_pattern, false);

        function = std::make_shared<ngraph::Function>(final_reshape, input_parameter, "TI_unrolled_without_memory");
    }

    void MemoryLSTMCellTest::Run() {
        SKIP_IF_CURRENT_TEST_IS_DISABLED()

        LoadNetwork();
        auto states = executableNetwork.QueryState();
        for (auto& state : states) {
            auto name = state.GetName();
            if (name == "cell_memory") {
                auto blob = FuncTestUtils::createAndFillBlobWithFloatArray(state.GetLastState()->getTensorDesc(),
                                                                           cell_memory_init.data(), cell_memory_init.size());
                state.SetState(blob);
            } else if (name == "hidden_memory") {
                auto blob = FuncTestUtils::createAndFillBlobWithFloatArray(state.GetLastState()->getTensorDesc(),
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

    TEST_P(MemoryLSTMCellTest, CompareWithRefs) {
        Run();
    };
}  // namespace SubgraphTestsDefinitions
