// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <transformations/control_flow/unroll_tensor_iterator.hpp>
#include "shared_test_classes/single_layer/tensor_iterator.hpp"

namespace LayerTestsDefinitions {

    std::string TensorIteratorTest::getTestCaseName(const testing::TestParamInfo<TensorIteratorParams> &obj) {
        bool should_decompose;
        size_t seq_lengths;
        size_t batch;
        size_t hidden_size;
        size_t input_size = 10;
        size_t sequence_axis;
        ngraph::helpers::TensorIteratorBody ti_body;
        float clip;
        ngraph::op::RecurrentSequenceDirection direction;
        InferenceEngine::Precision netPrecision;
        std::string targetDevice;
        std::tie(should_decompose, seq_lengths, batch, hidden_size, sequence_axis, clip, ti_body, direction, netPrecision,
                 targetDevice) = obj.param;
        std::vector<std::vector<size_t>> inputShapes = {};

        switch (ti_body) {
            case ngraph::helpers::TensorIteratorBody::LSTM:
                inputShapes = {
                        {{batch, input_size}, {batch, hidden_size}, {batch, hidden_size}, {4 * hidden_size, input_size},
                                {4 * hidden_size, hidden_size}, {4 * hidden_size}},
                };
                break;
            case ngraph::helpers::TensorIteratorBody::GRU:
                inputShapes = {
                        {{batch, input_size}, {batch, hidden_size}, {3 * hidden_size, input_size},
                                {3 * hidden_size, hidden_size}, {3 * hidden_size}},
                };
                break;
            case ngraph::helpers::TensorIteratorBody::RNN:
                inputShapes = {{batch, input_size}, {batch, hidden_size},
                               {hidden_size, input_size}, {hidden_size, hidden_size}, {hidden_size}};
                break;
        }

        std::ostringstream result;
        result << "unrolling=" << should_decompose << "_";
        result << "seq_len=" << seq_lengths << "_";
        result << "seq_len_axis=" << sequence_axis << "_";
        result << "batch=" << batch << "_";
        result << "hidden_size=" << hidden_size << "_";
        result << "input_size=" << input_size << "_";
        result << "IS=" << CommonTestUtils::vec2str(inputShapes) << "_";
        result << "TensorIteratorBody=" << ti_body << "_";
        result << "direction=" << direction << "_";
        result << "clip=" << clip << "_";
        result << "netPRC=" << netPrecision.name() << "_";
        result << "targetDevice=" << targetDevice << "_";
        return result.str();
    }

    void TensorIteratorTest::SetUp() {
        size_t seq_lengths;
        bool should_decompose;
        size_t batch;
        size_t hidden_size;
        size_t input_size = 10;
        size_t sequence_axis;
        ngraph::helpers::TensorIteratorBody ti_body;
        float clip;
        ngraph::op::RecurrentSequenceDirection direction;
        InferenceEngine::Precision netPrecision;
        std::tie(should_decompose, seq_lengths, batch, hidden_size, sequence_axis, clip, ti_body, direction, netPrecision,
                 targetDevice) = this->GetParam();
        std::vector<std::vector<size_t>> inputShapes;
        auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);
        auto tensor_iterator = std::make_shared<ngraph::opset5::TensorIterator>();

        // Each case consist of 3 steps:
        // 1. Create TensorIterator body.
        // 2. Set PortMap
        // 3. Create outer function
        auto axis = std::make_shared<ngraph::opset5::Constant>(ngraph::element::i64, ngraph::Shape{1},
                                                               std::vector<int64_t>{static_cast<int64_t>(sequence_axis)});
        switch (ti_body) {
            case ngraph::helpers::TensorIteratorBody::LSTM: {
                inputShapes = {
                        {{batch, seq_lengths, input_size}, {batch, hidden_size}, {batch, hidden_size}, {4 * hidden_size, input_size},
                                {4 * hidden_size, hidden_size}, {4 * hidden_size}},
                };
                if (sequence_axis == 0) {
                    // swap batch and seq_lengths
                    std::swap(inputShapes[0][0], inputShapes[0][1]);
                }
                auto outer_params = ngraph::builder::makeParams(ngPrc, {inputShapes[0], inputShapes[1], inputShapes[2]});

                // 1. Create TensorIterator body.
                inputShapes[0][sequence_axis] = 1; // sliced dimension
                auto body_params = ngraph::builder::makeParams(ngPrc, {inputShapes[0], inputShapes[1], inputShapes[2]});
                auto squeeze = std::make_shared<ngraph::opset5::Squeeze>(body_params[0], axis);
                std::vector<ngraph::Shape> WRB = {inputShapes[3], inputShapes[4], inputShapes[5]};
                ngraph::OutputVector out_vector = {squeeze, body_params[1], body_params[2]};
                auto lstm_cell = ngraph::builder::makeLSTM(out_vector, WRB, hidden_size, {"sigmoid", "tanh", "tanh"}, {}, {}, clip);
                auto unsqueeze = std::make_shared<ngraph::opset5::Unsqueeze>(lstm_cell->output(0), axis);
                ngraph::ResultVector results{std::make_shared<ngraph::opset1::Result>(unsqueeze),
                                             std::make_shared<ngraph::opset1::Result>(lstm_cell->output(0)),
                                             std::make_shared<ngraph::opset1::Result>(lstm_cell->output(1))};
                auto body = std::make_shared<ngraph::Function>(results, body_params, "lstm_cell");
                tensor_iterator->set_function(body);

                // 2. Set PortMap
                if (direction == ngraph::op::RecurrentSequenceDirection::FORWARD) {
                    tensor_iterator->set_sliced_input(body_params[0], outer_params[0], 0, 1, 1, -1, sequence_axis);
                    tensor_iterator->get_concatenated_slices(results[0], 0, 1, 1, -1, sequence_axis);
                } else if (direction == ngraph::op::RecurrentSequenceDirection::REVERSE) {
                    tensor_iterator->set_sliced_input(body_params[0], outer_params[0], -1, -1, 1, 0, sequence_axis);
                    tensor_iterator->get_concatenated_slices(results[0], -1, -1, 1, 0, sequence_axis);
                } else {
                    NGRAPH_CHECK(false, "Bidirectional case is not supported.");
                }

                tensor_iterator->set_merged_input(body_params[1], outer_params[1], results[1]);
                tensor_iterator->set_merged_input(body_params[2], outer_params[2], results[2]);
                tensor_iterator->get_iter_value(results[1]);
                tensor_iterator->get_iter_value(results[2]);

                // 3. Outer function
                function = std::make_shared<ngraph::Function>(ngraph::OutputVector{tensor_iterator->output(0), tensor_iterator->output(1),
                                                                                   tensor_iterator->output(2)}, outer_params);
                break;
            }
            case ngraph::helpers::TensorIteratorBody::GRU: {
                inputShapes = {
                        {{batch, seq_lengths, input_size}, {batch, hidden_size}, {3 * hidden_size, input_size},
                                {3 * hidden_size, hidden_size}, {3 * hidden_size}},
                };
                if (sequence_axis == 0) {
                    // swap batch and seq_lengths
                    std::swap(inputShapes[0][0], inputShapes[0][1]);
                }
                auto outer_params = ngraph::builder::makeParams(ngPrc, {inputShapes[0], inputShapes[1]});

                // 1. Create TensorIterator body.
                inputShapes[0][sequence_axis] = 1; // sliced dimension
                auto body_params = ngraph::builder::makeParams(ngPrc, {inputShapes[0], inputShapes[1]});
                std::vector<ngraph::Shape> WRB = {inputShapes[2], inputShapes[3], inputShapes[4]};
                auto squeeze = std::make_shared<ngraph::opset5::Squeeze>(body_params[0], axis);
                ngraph::OutputVector out_vector = {squeeze, body_params[1]};
                auto gru_cell = ngraph::builder::makeGRU(out_vector, WRB, hidden_size, {"sigmoid", "tanh"},
                                                         {}, {}, clip, false);
                auto unsqueeze = std::make_shared<ngraph::opset5::Unsqueeze>(gru_cell->output(0), axis);
                ngraph::ResultVector results{std::make_shared<ngraph::opset1::Result>(gru_cell->output(0)),
                                             std::make_shared<ngraph::opset1::Result>(unsqueeze)};
                auto body = std::make_shared<ngraph::Function>(results, body_params, "gru_cell");
                tensor_iterator->set_function(body);

                // 2. Set PortMap
                if (direction == ngraph::op::RecurrentSequenceDirection::FORWARD) {
                    tensor_iterator->set_sliced_input(body_params[0], outer_params[0], 0, 1, 1, -1, sequence_axis);
                    tensor_iterator->get_concatenated_slices(results[1], 0, 1, 1, -1, sequence_axis);
                } else if (direction == ngraph::op::RecurrentSequenceDirection::REVERSE) {
                    tensor_iterator->set_sliced_input(body_params[0], outer_params[0], -1, -1, 1, 0, sequence_axis);
                    tensor_iterator->get_concatenated_slices(results[1], -1, -1, 1, 0, sequence_axis);
                } else {
                    NGRAPH_CHECK(false, "Bidirectional case is not supported.");
                }

                tensor_iterator->set_merged_input(body_params[1], outer_params[1], results[0]);
                tensor_iterator->get_iter_value(results[0]);

                // 3. Outer function
                function = std::make_shared<ngraph::Function>(ngraph::OutputVector{tensor_iterator->output(0), tensor_iterator->output(1)}, outer_params);
                break;
            }
            case ngraph::helpers::TensorIteratorBody::RNN: {
                inputShapes = {{batch, seq_lengths, input_size},
                               {batch,       hidden_size},
                               {hidden_size, input_size},
                               {hidden_size, hidden_size},
                               {hidden_size}};
                if (sequence_axis == 0) {
                    // swap batch and seq_lengths
                    std::swap(inputShapes[0][0], inputShapes[0][1]);
                }
                auto outer_params = ngraph::builder::makeParams(ngPrc, {inputShapes[0], inputShapes[1]});

                // 1. Create TensorIterator body.
                inputShapes[0][sequence_axis] = 1; // sliced dimension
                auto body_params = ngraph::builder::makeParams(ngPrc, {inputShapes[0], inputShapes[1]});
                std::vector<ngraph::Shape> WRB = {inputShapes[2], inputShapes[3], inputShapes[4]};
                auto squeeze = std::make_shared<ngraph::opset5::Squeeze>(body_params[0], axis);
                ngraph::OutputVector out_vector = {squeeze, body_params[1]};
                auto rnn_cell = ngraph::builder::makeRNN(out_vector, WRB, hidden_size, {"tanh"}, {}, {}, clip);
                auto unsqueeze = std::make_shared<ngraph::opset5::Unsqueeze>(rnn_cell->output(0), axis);
                ngraph::ResultVector results{std::make_shared<ngraph::opset1::Result>(rnn_cell),
                                             std::make_shared<ngraph::opset1::Result>(unsqueeze)};
                auto body = std::make_shared<ngraph::Function>(results, body_params, "rnn_cell");
                tensor_iterator->set_function(body);

                // 2. Set PortMap
                if (direction == ngraph::op::RecurrentSequenceDirection::FORWARD) {
                    tensor_iterator->set_sliced_input(body_params[0], outer_params[0], 0, 1, 1, -1, sequence_axis);
                    tensor_iterator->get_concatenated_slices(results[1], 0, 1, 1, -1, sequence_axis);
                } else if (direction == ngraph::op::RecurrentSequenceDirection::REVERSE) {
                    tensor_iterator->set_sliced_input(body_params[0], outer_params[0], -1, -1, 1, 0, sequence_axis);
                    tensor_iterator->get_concatenated_slices(results[1], -1, -1, 1, 0, sequence_axis);
                } else {
                    NGRAPH_CHECK(false, "Bidirectional case is not supported.");
                }

                tensor_iterator->set_merged_input(body_params[1], outer_params[1], results[0]);
                tensor_iterator->get_iter_value(results[0]);

                // 3. Outer function
                function = std::make_shared<ngraph::Function>(ngraph::OutputVector{tensor_iterator->output(0), tensor_iterator->output(1)}, outer_params);
                break;
            }
        }
        if (should_decompose) {
            ngraph::pass::Manager m;
            m.register_pass<ngraph::pass::UnrollTensorIterator>();
            m.run_passes(function);
        }
    }
}  // namespace LayerTestsDefinitions
