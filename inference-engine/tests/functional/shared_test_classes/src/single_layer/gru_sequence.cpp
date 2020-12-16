// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/single_layer/gru_sequence.hpp"
#include "transformations/op_conversions/bidirectional_sequences_decomposition.hpp"
#include "transformations/op_conversions/convert_sequences_to_tensor_iterator.hpp"

namespace LayerTestsDefinitions {

    std::string GRUSequenceTest::getTestCaseName(const testing::TestParamInfo<GRUSequenceParams> &obj) {
        ngraph::helpers::SequenceTestsMode mode;
        size_t seq_lenghts;
        size_t batch;
        size_t hidden_size;
        size_t input_size = 10;
        std::vector<std::string> activations;
        std::vector<float> activations_alpha;
        std::vector<float> activations_beta;
        float clip;
        bool linear_before_reset;
        ngraph::op::RecurrentSequenceDirection direction;
        InferenceEngine::Precision netPrecision;
        std::string targetDevice;
        std::tie(mode, seq_lenghts, batch, hidden_size, activations, clip, linear_before_reset, direction, netPrecision,
                 targetDevice) = obj.param;
        std::vector<std::vector<size_t>> inputShapes = {
                {{batch, input_size}, {batch, hidden_size}, {batch, hidden_size}, {3 * hidden_size, input_size},
                        {3 * hidden_size, hidden_size}, {(linear_before_reset ? 4 : 3) * hidden_size}},
        };
        std::ostringstream result;
        result << "mode=" << mode << "_";
        result << "seq_lenghts=" << seq_lenghts << "_";
        result << "batch=" << batch << "_";
        result << "hidden_size=" << hidden_size << "_";
        result << "input_size=" << input_size << "_";
        result << "IS=" << CommonTestUtils::vec2str(inputShapes) << "_";
        result << "activations=" << CommonTestUtils::vec2str(activations) << "_";
        result << "direction=" << direction << "_";
        result << "clip=" << clip << "_";
        result << "netPRC=" << netPrecision.name() << "_";
        result << "targetDevice=" << targetDevice << "_";
        return result.str();
    }

    void GRUSequenceTest::SetUp() {
        size_t seq_lenghts;
        size_t batch;
        size_t hidden_size;
        size_t input_size = 10;
        std::vector<std::string> activations;
        std::vector<float> activations_alpha;
        std::vector<float> activations_beta;
        float clip;
        bool linear_before_reset;
        ngraph::op::RecurrentSequenceDirection direction;
        InferenceEngine::Precision netPrecision;
        std::tie(m_mode, seq_lenghts, batch, hidden_size, activations, clip, linear_before_reset, direction, netPrecision,
                 targetDevice) = this->GetParam();
        size_t num_directions = direction == ngraph::op::RecurrentSequenceDirection::BIDIRECTIONAL ? 2 : 1;
        std::vector<std::vector<size_t>> inputShapes = {
                {{batch, seq_lenghts, input_size}, {batch, num_directions, hidden_size}, {batch},
                 {num_directions, 3 * hidden_size, input_size}, {num_directions, 3 * hidden_size, hidden_size},
                 {num_directions, (linear_before_reset ? 4 : 3) * hidden_size}},
        };
        m_max_seq_len = seq_lenghts;
        auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);
        auto params = ngraph::builder::makeParams(ngPrc, {inputShapes[0], inputShapes[1]});
        if (m_mode == ngraph::helpers::SequenceTestsMode::CONVERT_TO_TI_MAX_SEQ_LEN_PARAM ||
            m_mode == ngraph::helpers::SequenceTestsMode::CONVERT_TO_TI_RAND_SEQ_LEN_PARAM) {
            auto seq_lengths = ngraph::builder::makeParams(ngraph::element::i64, {inputShapes[2]}).at(0);
            seq_lengths->set_friendly_name("seq_lengths");
            params.push_back(seq_lengths);
        }
        std::vector<ngraph::Shape> WRB = {inputShapes[3], inputShapes[4], inputShapes[5], inputShapes[2]};
        auto gru_sequence = ngraph::builder::makeGRU(ngraph::helpers::convert2OutputVector(ngraph::helpers::castOps2Nodes(params)),
                                                       WRB, hidden_size, activations, {}, {}, clip, linear_before_reset, true, direction,
                                                       m_mode);
        ngraph::ResultVector results{std::make_shared<ngraph::opset1::Result>(gru_sequence->output(0)),
                                     std::make_shared<ngraph::opset1::Result>(gru_sequence->output(1))};
        function = std::make_shared<ngraph::Function>(results, params, "gru_sequence");
        if (m_mode != ngraph::helpers::SequenceTestsMode::PURE_SEQ) {
            ngraph::pass::Manager manager;
            if (direction == ngraph::op::RecurrentSequenceDirection::BIDIRECTIONAL)
                manager.register_pass<ngraph::pass::BidirectionalGRUSequenceDecomposition>();
            manager.register_pass<ngraph::pass::ConvertGRUSequenceToTensorIterator>();
            manager.run_passes(function);
            bool ti_found = ngraph::helpers::is_tensor_iterator_exist(function);
            EXPECT_EQ(ti_found, true);
        } else {
            bool ti_found = ngraph::helpers::is_tensor_iterator_exist(function);
            EXPECT_EQ(ti_found, false);
        }
    }

    void GRUSequenceTest::Infer() {
        inferRequest = executableNetwork.CreateInferRequest();
        inputs.clear();

        for (const auto &input : executableNetwork.GetInputsInfo()) {
            const auto &info = input.second;
            auto blob = GenerateInput(*info);
            if (input.first == "seq_lengths") {
                blob = FuncTestUtils::createAndFillBlob(info->getTensorDesc(), m_max_seq_len, 0);
            }

            inferRequest.SetBlob(info->name(), blob);
            inputs.push_back(blob);
        }
        if (configuration.count(InferenceEngine::PluginConfigParams::KEY_DYN_BATCH_ENABLED) &&
            configuration.count(InferenceEngine::PluginConfigParams::YES)) {
            auto batchSize = executableNetwork.GetInputsInfo().begin()->second->getTensorDesc().getDims()[0] / 2;
            inferRequest.SetBatch(batchSize);
        }
        inferRequest.Infer();
    }
}  // namespace LayerTestsDefinitions
