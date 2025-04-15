// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <common_test_utils/ov_tensor_utils.hpp>
#include "transformations/control_flow/unroll_tensor_iterator.hpp"
#include "shared_test_classes/single_op/tensor_iterator.hpp"
#include "openvino/pass/manager.hpp"
#include "common_test_utils/node_builders/lstm_cell.hpp"
#include "common_test_utils/node_builders/gru_cell.hpp"
#include "common_test_utils/node_builders/rnn_cell.hpp"

namespace ov {
namespace test {
std::string TensorIteratorTest::getTestCaseName(const testing::TestParamInfo<TensorIteratorParams> &obj) {
    bool should_decompose;
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
    std::tie(should_decompose, seq_lengths, batch, hidden_size, sequence_axis, clip, ti_body, direction, model_type,
                target_device) = obj.param;
    std::vector<ov::Shape> input_shapes = {};

    switch (ti_body) {
        case ov::test::utils::TensorIteratorBody::LSTM:
            input_shapes = {
                    {{batch, input_size}, {batch, hidden_size}, {batch, hidden_size}, {4 * hidden_size, input_size},
                            {4 * hidden_size, hidden_size}, {4 * hidden_size}},
            };
            break;
        case ov::test::utils::TensorIteratorBody::GRU:
            input_shapes = {
                    {{batch, input_size}, {batch, hidden_size}, {3 * hidden_size, input_size},
                            {3 * hidden_size, hidden_size}, {3 * hidden_size}},
            };
            break;
        case ov::test::utils::TensorIteratorBody::RNN:
            input_shapes = {{batch, input_size}, {batch, hidden_size},
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
    result << "IS=" << ov::test::utils::vec2str(input_shapes) << "_";
    result << "TensorIteratorBody=" << ti_body << "_";
    result << "direction=" << direction << "_";
    result << "clip=" << clip << "_";
    result << "modelType=" << model_type.to_string() << "_";
    result << "targetDevice=" << target_device << "_";
    return result.str();
}

void TensorIteratorTest::SetUp() {
    size_t seq_lengths;
    bool should_decompose;
    size_t batch;
    size_t hidden_size;
    size_t input_size = 10;
    size_t sequence_axis;
    ov::test::utils::TensorIteratorBody ti_body;
    float clip;
    ov::op::RecurrentSequenceDirection direction;
    ov::element::Type model_type;
    std::tie(should_decompose, seq_lengths, batch, hidden_size, sequence_axis, clip, ti_body, direction, model_type,
                targetDevice) = this->GetParam();
    std::vector<ov::Shape> input_shapes;
    auto tensor_iterator = std::make_shared<ov::op::v0::TensorIterator>();

    // Each case consist of 3 steps:
    // 1. Create TensorIterator body.
    // 2. Set PortMap
    // 3. Create outer function
    auto axis = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{1},
                                                       std::vector<int64_t>{static_cast<int64_t>(sequence_axis)});
    switch (ti_body) {
        case ov::test::utils::TensorIteratorBody::LSTM: {
            input_shapes = {
                    {{batch, seq_lengths, input_size}, {batch, hidden_size}, {batch, hidden_size}, {4 * hidden_size, input_size},
                            {4 * hidden_size, hidden_size}, {4 * hidden_size}},
            };
            if (sequence_axis == 0) {
                // swap batch and seq_lengths
                std::swap(input_shapes[0][0], input_shapes[0][1]);
            }
            init_input_shapes(static_shapes_to_test_representation(input_shapes));
            ov::ParameterVector outer_params{std::make_shared<ov::op::v0::Parameter>(model_type, inputDynamicShapes[0]),
                                                std::make_shared<ov::op::v0::Parameter>(model_type, inputDynamicShapes[1]),
                                                std::make_shared<ov::op::v0::Parameter>(model_type, inputDynamicShapes[2])};

            // 1. Create TensorIterator body.
            inputDynamicShapes[0][sequence_axis] = 1; // sliced dimension
            ov::ParameterVector body_params{std::make_shared<ov::op::v0::Parameter>(model_type, inputDynamicShapes[0]),
                                            std::make_shared<ov::op::v0::Parameter>(model_type, inputDynamicShapes[1]),
                                            std::make_shared<ov::op::v0::Parameter>(model_type, inputDynamicShapes[2])};

            auto squeeze = std::make_shared<ov::op::v0::Squeeze>(body_params[0], axis);
            std::vector<ov::Shape> WRB = {input_shapes[3], input_shapes[4], input_shapes[5]};
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

            // 3. Outer model
            function = std::make_shared<ov::Model>(tensor_iterator->outputs(), outer_params);
            break;
        }
        case ov::test::utils::TensorIteratorBody::GRU: {
            input_shapes = {
                    {{batch, seq_lengths, input_size}, {batch, hidden_size}, {3 * hidden_size, input_size},
                            {3 * hidden_size, hidden_size}, {3 * hidden_size}},
            };
            if (sequence_axis == 0) {
                // swap batch and seq_lengths
                std::swap(input_shapes[0][0], input_shapes[0][1]);
            }
            init_input_shapes(static_shapes_to_test_representation(input_shapes));
            ov::ParameterVector outer_params{std::make_shared<ov::op::v0::Parameter>(model_type, inputDynamicShapes[0]),
                                             std::make_shared<ov::op::v0::Parameter>(model_type, inputDynamicShapes[1])};

            // 1. Create TensorIterator body.
            inputDynamicShapes[0][sequence_axis] = 1; // sliced dimension
            ov::ParameterVector body_params{std::make_shared<ov::op::v0::Parameter>(model_type, inputDynamicShapes[0]),
                                            std::make_shared<ov::op::v0::Parameter>(model_type, inputDynamicShapes[1])};

            std::vector<ov::Shape> WRB = {input_shapes[2], input_shapes[3], input_shapes[4]};
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
            input_shapes = {{batch, seq_lengths, input_size},
                            {batch,       hidden_size},
                            {hidden_size, input_size},
                            {hidden_size, hidden_size},
                            {hidden_size}};
            if (sequence_axis == 0) {
                // swap batch and seq_lengths
                std::swap(input_shapes[0][0], input_shapes[0][1]);
            }
            init_input_shapes(static_shapes_to_test_representation(input_shapes));
            ov::ParameterVector outer_params{std::make_shared<ov::op::v0::Parameter>(model_type, inputDynamicShapes[0]),
                                             std::make_shared<ov::op::v0::Parameter>(model_type, inputDynamicShapes[1])};

            // 1. Create TensorIterator body.
            inputDynamicShapes[0][sequence_axis] = 1; // sliced dimension
            ov::ParameterVector body_params{std::make_shared<ov::op::v0::Parameter>(model_type, inputDynamicShapes[0]),
                                            std::make_shared<ov::op::v0::Parameter>(model_type, inputDynamicShapes[1])};
            std::vector<ov::Shape> WRB = {input_shapes[2], input_shapes[3], input_shapes[4]};
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
    if (should_decompose) {
        ov::pass::Manager m;
        m.register_pass<ov::pass::UnrollTensorIterator>();
        m.run_passes(function);
    }
}

void TensorIteratorTest::generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) {
    inputs.clear();

    const auto& funcInputs = function->inputs();
    for (size_t i = 0; i < funcInputs.size(); i++) {
        const auto& funcInput = funcInputs[i];
        ov::test::utils::InputGenerateData in_data;
        in_data.start_from = 0;
        in_data.range = 8;
        in_data.resolution = funcInput.get_element_type().is_real() ? 32 : 1;
        ov::Tensor tensor = ov::test::utils::create_and_fill_tensor(funcInput.get_element_type(), targetInputStaticShapes[i], in_data);
        inputs.insert({funcInput.get_node_shared_ptr(), tensor});
    }
}

}  // namespace test
}  // namespace ov
