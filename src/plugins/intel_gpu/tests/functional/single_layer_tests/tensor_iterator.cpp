// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

//
#include "common_test_utils/ov_tensor_utils.hpp"
#include "common_test_utils/file_utils.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"
#include "common_test_utils/node_builders/convolution.hpp"
#include "common_test_utils/test_enums.hpp"
#include "common_test_utils/node_builders/gru_cell.hpp"
#include "common_test_utils/node_builders/lstm_cell.hpp"
#include "common_test_utils/node_builders/rnn_cell.hpp"

namespace {
using TensorIteratorWithConfigParams = typename std::tuple<
        size_t,                                 // seq_lengths
        size_t,                                 // batch
        size_t,                                 // hidden size
        // todo: fix. input size hardcoded to 10 due to limitation (10 args) of gtests Combine() func.
        //size_t,                               // input size
        size_t,                                 // sequence axis
        float,                                  // clip
        ov::test::utils::TensorIteratorBody,    // body type
        ov::op::RecurrentSequenceDirection,     // direction
        ov::element::Type,                      // Model type
        std::string>;                           // Device

class TensorIteratorWithConfigTest : public testing::WithParamInterface<TensorIteratorWithConfigParams>,
                                     virtual public ov::test::SubgraphBaseStaticTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<TensorIteratorWithConfigParams> &obj) {
        size_t seq_lengths;
        size_t batch;
        size_t hidden_size;
        size_t input_size = 10;
        size_t sequence_axis;
        ov::test::utils::TensorIteratorBody ti_body;
        float clip;
        ov::op::RecurrentSequenceDirection direction;
        ov::element::Type model_type;
        std::string target_device;
        std::tie(seq_lengths, batch, hidden_size, sequence_axis, clip, ti_body, direction, model_type, target_device) = obj.param;
        std::vector<std::vector<size_t>> inputShapes = {};

        switch (ti_body) {
            case ov::test::utils::TensorIteratorBody::LSTM:
                inputShapes = {
                        {{batch, input_size}, {batch, hidden_size}, {batch, hidden_size}, {4 * hidden_size, input_size},
                                {4 * hidden_size, hidden_size}, {4 * hidden_size}},
                };
                break;
            case ov::test::utils::TensorIteratorBody::GRU:
                inputShapes = {
                        {{batch, input_size}, {batch, hidden_size}, {3 * hidden_size, input_size},
                                {3 * hidden_size, hidden_size}, {3 * hidden_size}},
                };
                break;
            case ov::test::utils::TensorIteratorBody::RNN:
                inputShapes = {{batch, input_size}, {batch, hidden_size},
                               {hidden_size, input_size}, {hidden_size, hidden_size}, {hidden_size}};
                break;
        }

        std::ostringstream result;
        result << "seq_len=" << seq_lengths << "_";
        result << "seq_len_axis=" << sequence_axis << "_";
        result << "batch=" << batch << "_";
        result << "hidden_size=" << hidden_size << "_";
        result << "input_size=" << input_size << "_";
        result << "IS=" << ov::test::utils::vec2str(inputShapes) << "_";
        result << "TensorIteratorBody=" << ti_body << "_";
        result << "direction=" << direction << "_";
        result << "clip=" << clip << "_";
        result << "netPRC=" << model_type << "_";
        result << "targetDevice=" << target_device;
        return result.str();
    }

protected:
    void SetUp() override {
        size_t seq_lengths;
        size_t batch;
        size_t hidden_size;
        size_t input_size = 10;
        size_t sequence_axis;
        ov::test::utils::TensorIteratorBody ti_body;
        float clip;
        ov::op::RecurrentSequenceDirection direction;
        ov::element::Type model_type;
        std::tie(seq_lengths, batch, hidden_size, sequence_axis, clip, ti_body, direction, model_type, targetDevice) = this->GetParam();

        std::vector<std::vector<size_t>> inputShapes;
        auto tensor_iterator = std::make_shared<ov::op::v0::TensorIterator>();

        // Each case consist of 3 steps:
        // 1. Create TensorIterator body.
        // 2. Set PortMap
        // 3. Create outer function
        auto axis = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{1},
                                                               std::vector<int64_t>{static_cast<int64_t>(sequence_axis)});
        switch (ti_body) {
            case ov::test::utils::TensorIteratorBody::LSTM: {
                inputShapes = {
                        {{batch, seq_lengths, input_size}, {batch, hidden_size}, {batch, hidden_size}, {4 * hidden_size, input_size},
                                {4 * hidden_size, hidden_size}, {4 * hidden_size}},
                };
                if (sequence_axis == 0) {
                    // swap batch and seq_lengths
                    std::swap(inputShapes[0][0], inputShapes[0][1]);
                }
                ov::ParameterVector outer_params{std::make_shared<ov::op::v0::Parameter>(model_type, ov::Shape(inputShapes[0])),
                                                 std::make_shared<ov::op::v0::Parameter>(model_type, ov::Shape(inputShapes[1])),
                                                 std::make_shared<ov::op::v0::Parameter>(model_type, ov::Shape(inputShapes[2]))};

                // 1. Create TensorIterator body.
                inputShapes[0][sequence_axis] = 1; // sliced dimension
                ov::ParameterVector body_params{std::make_shared<ov::op::v0::Parameter>(model_type, ov::Shape(inputShapes[0])),
                                                std::make_shared<ov::op::v0::Parameter>(model_type, ov::Shape(inputShapes[1])),
                                                std::make_shared<ov::op::v0::Parameter>(model_type, ov::Shape(inputShapes[2]))};
                auto squeeze = std::make_shared<ov::op::v0::Squeeze>(body_params[0], axis);
                std::vector<ov::Shape> WRB = {inputShapes[3], inputShapes[4], inputShapes[5]};
                ov::OutputVector out_vector = {squeeze, body_params[1], body_params[2]};
                auto lstm_cell = ov::test::utils::make_lstm(out_vector, WRB, hidden_size, {"sigmoid", "tanh", "tanh"}, {}, {}, clip);
                auto unsqueeze = std::make_shared<ov::op::v0::Unsqueeze>(lstm_cell->output(0), axis);
                ov::ResultVector results{std::make_shared<ov::op::v0::Result>(unsqueeze),
                                             std::make_shared<ov::op::v0::Result>(lstm_cell->output(0)),
                                             std::make_shared<ov::op::v0::Result>(lstm_cell->output(1))};
                auto body = std::make_shared<ov::Model>(results, body_params, "lstm_cell");
                tensor_iterator->set_function(body);

                // 2. Set PortMap
                if (direction == ov::op::RecurrentSequenceDirection::FORWARD) {
                    tensor_iterator->set_sliced_input(body_params[0], outer_params[0], 0, 1, 1, -1, sequence_axis);
                    tensor_iterator->get_concatenated_slices(results[0], 0, 1, 1, -1, sequence_axis);
                } else if (direction == ov::op::RecurrentSequenceDirection::REVERSE) {
                    tensor_iterator->set_sliced_input(body_params[0], outer_params[0], -1, -1, 1, 0, sequence_axis);
                    tensor_iterator->get_concatenated_slices(results[0], -1, -1, 1, 0, sequence_axis);
                } else {
                    OPENVINO_THROW("Bidirectional case is not supported.");
                }

                tensor_iterator->set_merged_input(body_params[1], outer_params[1], results[1]);
                tensor_iterator->set_merged_input(body_params[2], outer_params[2], results[2]);
                tensor_iterator->get_iter_value(results[1]);
                tensor_iterator->get_iter_value(results[2]);

                // 3. Outer function
                function = std::make_shared<ov::Model>(ov::OutputVector{tensor_iterator->output(0), tensor_iterator->output(1),
                                                                                   tensor_iterator->output(2)}, outer_params);
                break;
            }
            case ov::test::utils::TensorIteratorBody::GRU: {
                inputShapes = {
                        {{batch, seq_lengths, input_size}, {batch, hidden_size}, {3 * hidden_size, input_size},
                                {3 * hidden_size, hidden_size}, {3 * hidden_size}},
                };
                if (sequence_axis == 0) {
                    // swap batch and seq_lengths
                    std::swap(inputShapes[0][0], inputShapes[0][1]);
                }
                ov::ParameterVector outer_params{std::make_shared<ov::op::v0::Parameter>(model_type, ov::Shape(inputShapes[0])),
                                                 std::make_shared<ov::op::v0::Parameter>(model_type, ov::Shape(inputShapes[1]))};

                // 1. Create TensorIterator body.
                inputShapes[0][sequence_axis] = 1; // sliced dimension
                ov::ParameterVector body_params{std::make_shared<ov::op::v0::Parameter>(model_type, ov::Shape(inputShapes[0])),
                                                std::make_shared<ov::op::v0::Parameter>(model_type, ov::Shape(inputShapes[1]))};
                std::vector<ov::Shape> WRB = {inputShapes[2], inputShapes[3], inputShapes[4]};
                auto squeeze = std::make_shared<ov::op::v0::Squeeze>(body_params[0], axis);
                ov::OutputVector out_vector = {squeeze, body_params[1]};
                auto gru_cell = ov::test::utils::make_gru(out_vector, WRB, hidden_size, {"sigmoid", "tanh"},
                                                          {}, {}, clip, false);
                auto unsqueeze = std::make_shared<ov::op::v0::Unsqueeze>(gru_cell->output(0), axis);
                ov::ResultVector results{std::make_shared<ov::op::v0::Result>(gru_cell->output(0)),
                                             std::make_shared<ov::op::v0::Result>(unsqueeze)};
                auto body = std::make_shared<ov::Model>(results, body_params, "gru_cell");
                tensor_iterator->set_function(body);

                // 2. Set PortMap
                if (direction == ov::op::RecurrentSequenceDirection::FORWARD) {
                    tensor_iterator->set_sliced_input(body_params[0], outer_params[0], 0, 1, 1, -1, sequence_axis);
                    tensor_iterator->get_concatenated_slices(results[1], 0, 1, 1, -1, sequence_axis);
                } else if (direction == ov::op::RecurrentSequenceDirection::REVERSE) {
                    tensor_iterator->set_sliced_input(body_params[0], outer_params[0], -1, -1, 1, 0, sequence_axis);
                    tensor_iterator->get_concatenated_slices(results[1], -1, -1, 1, 0, sequence_axis);
                } else {
                    OPENVINO_THROW("Bidirectional case is not supported.");
                }

                tensor_iterator->set_merged_input(body_params[1], outer_params[1], results[0]);
                tensor_iterator->get_iter_value(results[0]);

                // 3. Outer function
                function = std::make_shared<ov::Model>(ov::OutputVector{tensor_iterator->output(0), tensor_iterator->output(1)}, outer_params);
                break;
            }
            case ov::test::utils::TensorIteratorBody::RNN: {
                inputShapes = {{batch, seq_lengths, input_size},
                               {batch,       hidden_size},
                               {hidden_size, input_size},
                               {hidden_size, hidden_size},
                               {hidden_size}};
                if (sequence_axis == 0) {
                    // swap batch and seq_lengths
                    std::swap(inputShapes[0][0], inputShapes[0][1]);
                }
                ov::ParameterVector outer_params{std::make_shared<ov::op::v0::Parameter>(model_type, ov::Shape(inputShapes[0])),
                                                 std::make_shared<ov::op::v0::Parameter>(model_type, ov::Shape(inputShapes[1]))};

                // 1. Create TensorIterator body.
                inputShapes[0][sequence_axis] = 1; // sliced dimension
                ov::ParameterVector body_params{std::make_shared<ov::op::v0::Parameter>(model_type, ov::Shape(inputShapes[0])),
                                                std::make_shared<ov::op::v0::Parameter>(model_type, ov::Shape(inputShapes[1]))};
                std::vector<ov::Shape> WRB = {inputShapes[2], inputShapes[3], inputShapes[4]};
                auto squeeze = std::make_shared<ov::op::v0::Squeeze>(body_params[0], axis);
                ov::OutputVector out_vector = {squeeze, body_params[1]};
                auto rnn_cell = ov::test::utils::make_rnn(out_vector, WRB, hidden_size, {"tanh"}, {}, {}, clip);
                auto unsqueeze = std::make_shared<ov::op::v0::Unsqueeze>(rnn_cell->output(0), axis);
                ov::ResultVector results{std::make_shared<ov::op::v0::Result>(rnn_cell),
                                             std::make_shared<ov::op::v0::Result>(unsqueeze)};
                auto body = std::make_shared<ov::Model>(results, body_params, "rnn_cell");
                tensor_iterator->set_function(body);

                // 2. Set PortMap
                if (direction == ov::op::RecurrentSequenceDirection::FORWARD) {
                    tensor_iterator->set_sliced_input(body_params[0], outer_params[0], 0, 1, 1, -1, sequence_axis);
                    tensor_iterator->get_concatenated_slices(results[1], 0, 1, 1, -1, sequence_axis);
                } else if (direction == ov::op::RecurrentSequenceDirection::REVERSE) {
                    tensor_iterator->set_sliced_input(body_params[0], outer_params[0], -1, -1, 1, 0, sequence_axis);
                    tensor_iterator->get_concatenated_slices(results[1], -1, -1, 1, 0, sequence_axis);
                } else {
                    OPENVINO_THROW("Bidirectional case is not supported.");
                }

                tensor_iterator->set_merged_input(body_params[1], outer_params[1], results[0]);
                tensor_iterator->get_iter_value(results[0]);

                // 3. Outer function
                function = std::make_shared<ov::Model>(ov::OutputVector{tensor_iterator->output(0), tensor_iterator->output(1)}, outer_params);
                break;
            }
        }
    }
};

TEST_P(TensorIteratorWithConfigTest, Inference) {
    run();
};

INSTANTIATE_TEST_SUITE_P(smoke_TensorIteratorCommon, TensorIteratorWithConfigTest,
    ::testing::Combine(
        ::testing::ValuesIn(std::vector<size_t> {2, 4}), // seq lengths
        ::testing::ValuesIn(std::vector<size_t> {1}), // only single batch supported
        ::testing::ValuesIn(std::vector<size_t> {2, 4}), // hidden size
        ::testing::ValuesIn(std::vector<size_t> {0, 1}), // seq axis
        ::testing::ValuesIn(std::vector<float> {0.f}), // clip - not used
        ::testing::ValuesIn(std::vector<ov::test::utils::TensorIteratorBody> {
            ov::test::utils::TensorIteratorBody::LSTM,
            ov::test::utils::TensorIteratorBody::RNN,
            ov::test::utils::TensorIteratorBody::GRU,
        }), // body type
        ::testing::ValuesIn(std::vector<ov::op::RecurrentSequenceDirection>{
            ov::op::RecurrentSequenceDirection::FORWARD,
            ov::op::RecurrentSequenceDirection::REVERSE,
        }),
        ::testing::ValuesIn(std::vector<ov::element::Type> {
            ov::element::f32,
            ov::element::f16,
        }),
        ::testing::Values(ov::test::utils::DEVICE_GPU)),
    TensorIteratorWithConfigTest::getTestCaseName);
}  // namespace
