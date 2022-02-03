// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/subgraph/basic_lstm.hpp"
#include <transformations/op_conversions/lstm_cell_decomposition.hpp>

using namespace SubgraphTestsDefinitions;

namespace {

class Basic_LSTM_SW_FP32 : public Basic_LSTM_S {
public:
    std::shared_ptr<ngraph::Function> GetNetwork_sw_fp32(size_t thirdDimOut,
                                                            size_t hiddenSize,
                                                            size_t num_cells,
                                                            const InferenceEngine::Precision& netPrecission,
                                                            std::vector<float>* hidden_memory_init_out,
                                                            std::vector<float>* cell_memory_init_out) {
        auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecission);

        auto params = ngraph::builder::makeParams(ngPrc, { {1, num_cells * thirdDimOut} });

        const size_t batch_size = 1;

        //Reshape_1 [1,thirdDimOut*num_cells] -> [1, num_cells, thirdDimOut]
        std::vector<uint64_t> outFormShapes1 = { batch_size, num_cells, thirdDimOut };
        auto pattern1 = std::make_shared<ngraph::opset1::Constant>(ngraph::element::Type_t::i64, ngraph::Shape{ 3 }, outFormShapes1);
        auto reshape1 = std::make_shared<ngraph::opset1::Reshape>(params[0], pattern1, false);

        auto reshape1_shape = reshape1->output(0).get_shape();
        auto H_init = ngraph::builder::makeConstant<float>(ngPrc, { batch_size, hiddenSize }, {}, true, 0.1f, 0.f);
        auto C_init = ngraph::builder::makeConstant<float>(ngPrc, { batch_size, hiddenSize }, {}, true, 0.1f, 0.f);
        if (hidden_memory_init_out != nullptr) {
            *hidden_memory_init_out = std::static_pointer_cast<ngraph::opset1::Constant>(H_init)->cast_vector<float>();
        }
        if (cell_memory_init_out != nullptr) {
            *cell_memory_init_out = std::static_pointer_cast<ngraph::opset1::Constant>(C_init)->cast_vector<float>();
        }
        auto H_t = std::make_shared<ngraph::opset1::Parameter>(ngPrc, ngraph::Shape{ batch_size, hiddenSize });
        auto C_t = std::make_shared<ngraph::opset1::Parameter>(ngPrc, ngraph::Shape{ batch_size, hiddenSize });
        H_t->set_friendly_name("hidden_state_1");
        C_t->set_friendly_name("cell_state_1");
        //Body
        auto X = std::make_shared<ngraph::opset1::Parameter>(ngPrc, ngraph::Shape{ batch_size, 1, reshape1_shape[2] });
        auto weightsNode = ngraph::builder::makeConstant<float>(ngPrc, { 4 * hiddenSize, reshape1_shape[2] }, {}, true, 0.05f, 0.f);
        auto reccurrenceWeightsNode = ngraph::builder::makeConstant<float>(ngPrc, { 4 * hiddenSize, hiddenSize }, {}, true, 0.05f, 0.f);

        //lstm [1, 10], [1, 118], [1, 118] -> [1, 118], [1, 118]
        outFormShapes1 = { batch_size, reshape1_shape[2] };
        auto constantX = std::make_shared<ngraph::opset1::Constant>(ngraph::element::i64, ngraph::Shape{ 2 }, outFormShapes1);
        auto lstm1 = std::make_shared<ngraph::opset4::LSTMCell>(std::make_shared<ngraph::opset1::Reshape>(X, constantX, false),
            H_t, C_t,
            weightsNode, reccurrenceWeightsNode, hiddenSize);

        auto H_o = lstm1->output(0);
        auto C_o = lstm1->output(1);

        //TensorIterator [1, num_cells, thirdDimOut] [1, 118], [1, 118] -> [1, 118]
        auto body = std::make_shared<ngraph::Function>(
            ngraph::OutputVector{ H_o, C_o }, ngraph::ParameterVector{ X, H_t, C_t });

        auto tensor_iterator = std::make_shared<ngraph::opset1::TensorIterator>();
        tensor_iterator->set_body(body);

        //input tensor shape: [1, num_cells, thirdDimOut] chunk shape: [1, 1, thirdDimOut]
        tensor_iterator->set_sliced_input(X, reshape1, 0, 1, 1, -1, 1);
        tensor_iterator->set_merged_input(H_t, H_init, H_o);
        tensor_iterator->set_merged_input(C_t, C_init, C_o);

        auto out0 = tensor_iterator->get_iter_value(H_o, -1);

        const size_t output_size = 12;
        auto fc1 = ngraph::builder::makeFullyConnected(out0, ngPrc, output_size, true, { hiddenSize, output_size }, { 0.05f }, { 0.1f });

        ngraph::ResultVector results{ std::make_shared<ngraph::opset1::Result>(fc1) };
        return std::make_shared<ngraph::Function>(results, params, "Basic_LSTM_S");
    }

protected:
    InferenceEngine::Blob::Ptr GenerateInput(const InferenceEngine::InputInfo &info) const override {
        return FuncTestUtils::createAndFillBlob(info.getTensorDesc(), 0.05f, 0.f);
    }

    void SetUp() override {
        threshold = 0.1f;

        InferenceEngine::Precision netPrecision;
        std::pair<size_t, size_t> size_params;
        size_t num_cells;
        bool decompose;
        std::tie(netPrecision, targetDevice, configuration, size_params, num_cells, decompose) = this->GetParam();
        third_dim = size_params.first;
        hidden_size = size_params.second;
        outPrc = InferenceEngine::Precision::FP32;

        function = GetNetwork_sw_fp32(size_params.first, size_params.second, num_cells, netPrecision, &hidden_memory_init, &cell_memory_init);
        if (decompose) {
            ngraph::pass::Manager manager;
            manager.register_pass<ngraph::pass::LSTMCellDecomposition>();
            manager.run_passes(function);
        }
    }
};

TEST_P(Basic_LSTM_SW_FP32, CompareWithRefImpl) {
    Run();
};

const std::vector<InferenceEngine::Precision> netPrecisions = {
        InferenceEngine::Precision::FP32,
        InferenceEngine::Precision::FP16
};

std::map<std::string, std::string> config_no_sf = {
    {"GNA_COMPACT_MODE", "NO"},
    {"GNA_DEVICE_MODE", "GNA_SW_FP32"}
};

std::pair<size_t, size_t> size_params = {49, 118};

const std::vector<bool> decompose = { false, true };

INSTANTIATE_TEST_SUITE_P(smoke_BasicLSTMSWFP32_no_scale_factor, Basic_LSTM_SW_FP32,
                        ::testing::Combine(
                            ::testing::ValuesIn(netPrecisions),
                            ::testing::Values(CommonTestUtils::DEVICE_GNA),
                            ::testing::Values(config_no_sf),
                            ::testing::Values(size_params),
                            ::testing::Values(3),
                            ::testing::ValuesIn(decompose)),
                        Basic_LSTM_S::getTestCaseName);

} // namespace