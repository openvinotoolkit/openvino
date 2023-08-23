// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/single_layer/lstm_sequence.hpp"
#include "transformations/op_conversions/bidirectional_sequences_decomposition.hpp"
#include "transformations/op_conversions/convert_sequences_to_tensor_iterator.hpp"
#include "ngraph/pass/visualize_tree.hpp"

namespace LayerTestsDefinitions {

    using ngraph::helpers::InputLayerType;

    std::string LSTMSequenceTest::getTestCaseName(const testing::TestParamInfo<LSTMSequenceParams> &obj) {
        ngraph::helpers::SequenceTestsMode mode;
        size_t seq_lengths;
        size_t batch;
        size_t hidden_size;
        size_t input_size;
        std::vector<std::string> activations;
        std::vector<float> activations_alpha;
        std::vector<float> activations_beta;
        float clip;
        ngraph::op::RecurrentSequenceDirection direction;
        InputLayerType WRBType;
        InferenceEngine::Precision netPrecision;
        std::string targetDevice;
        std::tie(mode, seq_lengths, batch, hidden_size, input_size, activations, clip, direction,
                 WRBType, netPrecision, targetDevice) = obj.param;
        std::vector<std::vector<size_t>> inputShapes = {
                {{batch, input_size}, {batch, hidden_size}, {batch, hidden_size}, {4 * hidden_size, input_size},
                        {4 * hidden_size, hidden_size}, {4 * hidden_size}},
        };
        std::ostringstream result;
        result << "mode=" << mode << "_";
        result << "seq_lengths=" << seq_lengths << "_";
        result << "batch=" << batch << "_";
        result << "hidden_size=" << hidden_size << "_";
        result << "input_size=" << input_size << "_";
        result << "IS=" << ov::test::utils::vec2str(inputShapes) << "_";
        result << "activations=" << ov::test::utils::vec2str(activations) << "_";
        result << "direction=" << direction << "_";
        result << "clip=" << clip << "_";
        result << "WRBType=" << WRBType << "_";
        result << "netPRC=" << netPrecision.name() << "_";
        result << "targetDevice=" << targetDevice << "_";
        return result.str();
    }

    void LSTMSequenceTest::SetUp() {
        using namespace ngraph::helpers;
        using namespace ngraph::builder;
        size_t seq_lengths;

        size_t batch;
        size_t hidden_size;
        size_t input_size;
        std::vector<std::string> activations;
        std::vector<float> activations_alpha;
        std::vector<float> activations_beta;
        float clip;
        ngraph::op::RecurrentSequenceDirection direction;
        InputLayerType WRBType;
        InferenceEngine::Precision netPrecision;
        std::tie(m_mode, seq_lengths, batch, hidden_size, input_size, activations, clip, direction,
                 WRBType, netPrecision, targetDevice) = this->GetParam();
        size_t num_directions = direction == ngraph::op::RecurrentSequenceDirection::BIDIRECTIONAL ? 2 : 1;
        m_max_seq_len = seq_lengths;
        std::vector<ov::Shape> inputShapes = {
                {{batch, seq_lengths, input_size}, {batch, num_directions, hidden_size}, {batch, num_directions, hidden_size},
                 {batch}, {num_directions, 4 * hidden_size, input_size}, {num_directions, 4 * hidden_size, hidden_size}, {num_directions, 4 * hidden_size}},
        };

        const auto& W_shape = inputShapes[4];
        const auto& R_shape = inputShapes[5];
        const auto& B_shape = inputShapes[6];

        auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);
        ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(ngPrc, inputShapes[0]),
                                   std::make_shared<ov::op::v0::Parameter>(ngPrc, inputShapes[1]),
                                   std::make_shared<ov::op::v0::Parameter>(ngPrc, inputShapes[2])};

        std::shared_ptr<ov::Node> seq_lengths_node;
        if (m_mode == SequenceTestsMode::CONVERT_TO_TI_MAX_SEQ_LEN_PARAM ||
            m_mode == SequenceTestsMode::CONVERT_TO_TI_RAND_SEQ_LEN_PARAM ||
            m_mode == SequenceTestsMode::PURE_SEQ_RAND_SEQ_LEN_PARAM) {
            auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, inputShapes[3]);
            seq_lengths_node = param;
            seq_lengths_node->set_friendly_name("seq_lengths");
            params.push_back(param);
        } else if (m_mode == ngraph::helpers::SequenceTestsMode::CONVERT_TO_TI_RAND_SEQ_LEN_CONST ||
                   m_mode == ngraph::helpers::SequenceTestsMode::PURE_SEQ_RAND_SEQ_LEN_CONST) {
            seq_lengths_node = ngraph::builder::makeConstant<int64_t>(ov::element::i64, inputShapes[3], {}, true,
                                                                      static_cast<int64_t>(seq_lengths), 0.f);
        } else {
            std::vector<int64_t> lengths(inputShapes[3][0], seq_lengths);
            seq_lengths_node = ngraph::builder::makeConstant(ov::element::i64, inputShapes[3], lengths, false);
        }

        std::shared_ptr<ov::Node> W, R, B;
        if (WRBType == InputLayerType::PARAMETER) {
            const auto W_param = std::make_shared<ov::op::v0::Parameter>(ngPrc, W_shape);
            const auto R_param = std::make_shared<ov::op::v0::Parameter>(ngPrc, R_shape);
            const auto B_param = std::make_shared<ov::op::v0::Parameter>(ngPrc, B_shape);
            W = W_param;
            R = R_param;
            B = B_param;
            params.push_back(W_param);
            params.push_back(R_param);
            params.push_back(B_param);
        } else {
            W = ngraph::builder::makeConstant<float>(ngPrc, W_shape, {}, true);
            R = ngraph::builder::makeConstant<float>(ngPrc, R_shape, {}, true);
            B = ngraph::builder::makeConstant<float>(ngPrc, B_shape, {}, true);
        }

        auto lstm_sequence = std::make_shared<ov::op::v5::LSTMSequence>(params[0], params[1], params[2], seq_lengths_node, W, R, B, hidden_size, direction,
                std::vector<float>{}, std::vector<float>{}, activations, clip);
        ngraph::ResultVector results{std::make_shared<ngraph::opset1::Result>(lstm_sequence->output(0)),
                                     std::make_shared<ngraph::opset1::Result>(lstm_sequence->output(1)),
                                     std::make_shared<ngraph::opset1::Result>(lstm_sequence->output(2))};
        function = std::make_shared<ngraph::Function>(results, params, "lstm_sequence");
        bool is_pure_sequence = (m_mode == SequenceTestsMode::PURE_SEQ ||
                                 m_mode == SequenceTestsMode::PURE_SEQ_RAND_SEQ_LEN_PARAM ||
                                 m_mode == SequenceTestsMode::PURE_SEQ_RAND_SEQ_LEN_CONST);
        if (!is_pure_sequence) {
            ngraph::pass::Manager manager;
            if (direction == ngraph::op::RecurrentSequenceDirection::BIDIRECTIONAL)
                manager.register_pass<ov::pass::BidirectionalLSTMSequenceDecomposition>();
            manager.register_pass<ov::pass::ConvertLSTMSequenceToTensorIterator>();
            manager.run_passes(function);
            bool ti_found = is_tensor_iterator_exist(function);
            EXPECT_EQ(ti_found, true);
        } else {
            bool ti_found = is_tensor_iterator_exist(function);
            EXPECT_EQ(ti_found, false);
        }
    }

    void LSTMSequenceTest::GenerateInputs() {
        for (const auto &input : executableNetwork.GetInputsInfo()) {
            const auto &info = input.second;
            auto blob = GenerateInput(*info);
            if (input.first == "seq_lengths") {
                blob = FuncTestUtils::createAndFillBlob(info->getTensorDesc(), m_max_seq_len, 0);
            }

            inputs.push_back(blob);
        }
    }
}  // namespace LayerTestsDefinitions
