// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <cmath>
#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "core/operator_set.hpp"
#include "exceptions.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/lstm_sequence.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/op/util/rnn_cell_base.hpp"
#include "openvino/util/common_util.hpp"
#include "utils/common.hpp"
#include "utils/reshape.hpp"
#include "utils/split.hpp"

namespace ov {
namespace frontend {
namespace onnx {
namespace com_microsoft {
namespace {

enum class DynamicQuantizeLSTMInput {
    X,
    W,
    R,
    B,
    SEQUENCE_LENS,
    INITIAL_H,
    INITIAL_C,
};

struct PreparedQuantizedWeights {
    ov::Output<ov::Node> weights;
    int64_t original_rank;
};

ov::Output<ov::Node> normalize_tensor_rank(const ov::Output<ov::Node>& input,
                                           int64_t target_rank,
                                           const std::string& input_name) {
    const auto& input_rank = input.get_partial_shape().rank();

    if (input_rank.is_dynamic()) {
        return input;
    }

    if (input_rank.get_length() == target_rank) {
        return input;
    }

    if (input_rank.get_length() > target_rank) {
        const auto dims_to_squeeze = input_rank.get_length() - target_rank;
        const auto& input_shape = input.get_partial_shape();

        for (int64_t i = 0; i < dims_to_squeeze; ++i) {
            if (input_shape[i].is_static() && input_shape[i].get_length() != 1) {
                OPENVINO_THROW("DynamicQuantizeLSTM input '",
                               input_name,
                               "' has rank ",
                               input_rank.get_length(),
                               " but expected ",
                               target_rank,
                               ". Leading dimension [",
                               i,
                               "] is ",
                               input_shape[i].get_length(),
                               " but must be 1 to squeeze.");
            }
        }

        std::vector<int64_t> axes_to_squeeze;
        axes_to_squeeze.reserve(dims_to_squeeze);
        for (int64_t i = 0; i < dims_to_squeeze; ++i) {
            axes_to_squeeze.push_back(i);
        }

        auto axes_const =
            ov::op::v0::Constant::create(ov::element::i64, ov::Shape{axes_to_squeeze.size()}, axes_to_squeeze);
        return std::make_shared<ov::op::v0::Squeeze>(input, axes_const);
    }

    const auto dims_to_unsqueeze = target_rank - input_rank.get_length();
    if (dims_to_unsqueeze != 1) {
        OPENVINO_THROW("DynamicQuantizeLSTM input '",
                       input_name,
                       "' has rank ",
                       input_rank.get_length(),
                       " but expected ",
                       target_rank,
                       ". Rank difference is ",
                       dims_to_unsqueeze,
                       " but only 1 is supported.");
    }

    auto axes_const = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {0});
    return std::make_shared<ov::op::v0::Unsqueeze>(input, axes_const);
}

PreparedQuantizedWeights prepare_quantized_weights(const ov::frontend::onnx::Node& node,
                                                   const ov::Output<ov::Node>& weights,
                                                   int64_t hidden_size,
                                                   const std::string& input_name) {
    const auto& rank = weights.get_partial_shape().rank();
    CHECK_VALID_NODE(node,
                     rank.is_static() && (rank.get_length() == 2 || rank.get_length() == 3),
                     "DynamicQuantizeLSTM input '",
                     input_name,
                     "' must have static rank 2 or 3. Got rank: ",
                     rank);

    const auto gate_axis_size = 4 * hidden_size;
    auto prepared_weights = weights;

    if (rank.get_length() == 2) {
        const auto& shape = prepared_weights.get_partial_shape();
        const auto dim0_matches = shape[0].is_static() && shape[0].get_length() == gate_axis_size;
        const auto dim1_matches = shape[1].is_static() && shape[1].get_length() == gate_axis_size;

        CHECK_VALID_NODE(node,
                         dim0_matches || dim1_matches,
                         "DynamicQuantizeLSTM input '",
                         input_name,
                         "' must have one dimension equal to 4*hidden_size (",
                         gate_axis_size,
                         "). Got shape: ",
                         shape);

        if (!dim0_matches && dim1_matches) {
            prepared_weights = ov::op::util::reorder_axes(prepared_weights, {1, 0});
        }

        auto axis = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {0});
        prepared_weights = std::make_shared<ov::op::v0::Unsqueeze>(prepared_weights, axis);
    } else {
        const auto& shape = prepared_weights.get_partial_shape();
        const auto dim1_matches = shape[1].is_static() && shape[1].get_length() == gate_axis_size;
        const auto dim2_matches = shape[2].is_static() && shape[2].get_length() == gate_axis_size;

        CHECK_VALID_NODE(node,
                         dim1_matches || dim2_matches,
                         "DynamicQuantizeLSTM input '",
                         input_name,
                         "' must have either axis 1 or axis 2 equal to 4*hidden_size (",
                         gate_axis_size,
                         "). Got shape: ",
                         shape);

        if (!dim1_matches && dim2_matches) {
            prepared_weights = ov::op::util::reorder_axes(prepared_weights, {0, 2, 1});
        }
    }

    return {prepared_weights, rank.get_length()};
}

ov::Output<ov::Node> align_dequant_param(const ov::frontend::onnx::Node& node,
                                         const ov::Output<ov::Node>& param,
                                         int64_t weight_rank,
                                         const std::string& input_name) {
    const auto& rank = param.get_partial_shape().rank();

    if (rank.is_dynamic() || rank.get_length() == 0) {
        return param;
    }

    CHECK_VALID_NODE(node,
                     rank.get_length() == 1 || rank.get_length() == 2,
                     "DynamicQuantizeLSTM input '",
                     input_name,
                     "' must be a scalar, rank-1, or rank-2 tensor. Got rank: ",
                     rank.get_length());

    if (weight_rank == 2) {
        CHECK_VALID_NODE(node,
                         rank.get_length() == 1,
                         "DynamicQuantizeLSTM rank-2 weights require rank-1 scale/zero_point for input '",
                         input_name,
                         "'.");
        auto axes = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{2}, {0, 2});
        return std::make_shared<ov::op::v0::Unsqueeze>(param, axes);
    }

    if (rank.get_length() == 1) {
        auto axes = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{2}, {1, 2});
        return std::make_shared<ov::op::v0::Unsqueeze>(param, axes);
    }

    auto axes = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {2});
    return std::make_shared<ov::op::v0::Unsqueeze>(param, axes);
}

ov::Output<ov::Node> dequantize_weights(const ov::frontend::onnx::Node& node,
                                        const PreparedQuantizedWeights& prepared_weights,
                                        const ov::Output<ov::Node>& scale,
                                        const ov::Output<ov::Node>& zero_point,
                                        const std::string& weights_name) {
    auto dequantized =
        ov::Output<ov::Node>(std::make_shared<ov::op::v0::Convert>(prepared_weights.weights, scale.get_element_type()));
    auto aligned_scale = align_dequant_param(node, scale, prepared_weights.original_rank, weights_name + "_scale");

    if (zero_point.get_node_shared_ptr()) {
        auto aligned_zero_point = align_dequant_param(node,
                                                      zero_point,
                                                      prepared_weights.original_rank,
                                                      weights_name + "_zero_point");
        aligned_zero_point = std::make_shared<ov::op::v0::Convert>(aligned_zero_point, scale.get_element_type());
        dequantized = std::make_shared<ov::op::v1::Subtract>(dequantized, aligned_zero_point);
    }

    return std::make_shared<ov::op::v1::Multiply>(dequantized, aligned_scale);
}

struct DynamicQuantizeLSTMInputMap {
    explicit DynamicQuantizeLSTMInputMap(const ov::frontend::onnx::Node& node, int64_t hidden_size) {
        const auto& ng_inputs = node.get_ov_inputs();
        constexpr std::size_t gates_count{4};

        auto input_x = normalize_tensor_rank(ng_inputs.at(0), 3, "X");
        m_input_map[DynamicQuantizeLSTMInput::X] = ov::op::util::reorder_axes(input_x, {1, 0, 2});

        CHECK_VALID_NODE(node,
                         ng_inputs.at(1).get_element_type() == ov::element::i8 ||
                             ng_inputs.at(1).get_element_type() == ov::element::u8,
                         "DynamicQuantizeLSTM input 'W' must have element type int8 or uint8. Got: ",
                         ng_inputs.at(1).get_element_type());
        CHECK_VALID_NODE(node,
                         ng_inputs.at(2).get_element_type() == ov::element::i8 ||
                             ng_inputs.at(2).get_element_type() == ov::element::u8,
                         "DynamicQuantizeLSTM input 'R' must have element type int8 or uint8. Got: ",
                         ng_inputs.at(2).get_element_type());
        CHECK_VALID_NODE(node,
                         ng_inputs.at(8).get_element_type() == ov::element::f32,
                         "DynamicQuantizeLSTM input 'W_scale' must have element type float32. Got: ",
                         ng_inputs.at(8).get_element_type());
        CHECK_VALID_NODE(node,
                         ng_inputs.at(10).get_element_type() == ov::element::f32,
                         "DynamicQuantizeLSTM input 'R_scale' must have element type float32. Got: ",
                         ng_inputs.at(10).get_element_type());

        ov::Output<ov::Node> w_zero_point;
        if (common::is_input_valid(node, 9)) {
            w_zero_point = ng_inputs.at(9);
            CHECK_VALID_NODE(node,
                             w_zero_point.get_element_type() == ov::element::i8 ||
                                 w_zero_point.get_element_type() == ov::element::u8,
                             "DynamicQuantizeLSTM input 'W_zero_point' must have element type int8 or uint8. Got: ",
                             w_zero_point.get_element_type());
        }

        ov::Output<ov::Node> r_zero_point;
        if (common::is_input_valid(node, 11)) {
            r_zero_point = ng_inputs.at(11);
            CHECK_VALID_NODE(node,
                             r_zero_point.get_element_type() == ov::element::i8 ||
                                 r_zero_point.get_element_type() == ov::element::u8,
                             "DynamicQuantizeLSTM input 'R_zero_point' must have element type int8 or uint8. Got: ",
                             r_zero_point.get_element_type());
        }

        auto prepared_w = prepare_quantized_weights(node, ng_inputs.at(1), hidden_size, "W");
        auto prepared_r = prepare_quantized_weights(node, ng_inputs.at(2), hidden_size, "R");

        m_input_map[DynamicQuantizeLSTMInput::W] =
            ov::op::util::convert_lstm_node_format(dequantize_weights(node, prepared_w, ng_inputs.at(8), w_zero_point, "W"),
                                                   ov::op::util::LSTMWeightsFormat::IOFC,
                                                   ov::op::util::LSTMWeightsFormat::FICO,
                                                   1);
        m_input_map[DynamicQuantizeLSTMInput::R] =
            ov::op::util::convert_lstm_node_format(dequantize_weights(node, prepared_r, ng_inputs.at(10), r_zero_point, "R"),
                                                   ov::op::util::LSTMWeightsFormat::IOFC,
                                                   ov::op::util::LSTMWeightsFormat::FICO,
                                                   1);

        auto shape_of_x = std::make_shared<ov::op::v3::ShapeOf>(m_input_map[DynamicQuantizeLSTMInput::X]);
        auto axes = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{1}, {0});
        auto batch_size_node = std::make_shared<ov::op::v8::Gather>(shape_of_x,
                                                                    ov::op::v0::Constant::create(ov::element::i32,
                                                                                                  ov::Shape{1},
                                                                                                  {0}),
                                                                    axes);
        auto seq_length_node = std::make_shared<ov::op::v8::Gather>(shape_of_x,
                                                                    ov::op::v0::Constant::create(ov::element::i32,
                                                                                                  ov::Shape{1},
                                                                                                  {1}),
                                                                    axes);

        auto shape_of_r = std::make_shared<ov::op::v3::ShapeOf>(m_input_map[DynamicQuantizeLSTMInput::R]);
        auto num_directions_node = std::make_shared<ov::op::v8::Gather>(shape_of_r,
                                                                        ov::op::v0::Constant::create(ov::element::i32,
                                                                                                      ov::Shape{1},
                                                                                                      {0}),
                                                                        axes);
        auto hidden_size_node = std::make_shared<ov::op::v8::Gather>(shape_of_r,
                                                                     ov::op::v0::Constant::create(ov::element::i32,
                                                                                                   ov::Shape{1},
                                                                                                   {2}),
                                                                     axes);

        if (common::is_input_valid(node, 3)) {
            auto bias = normalize_tensor_rank(ng_inputs.at(3), 2, "B");
            auto split_bias = ov::op::util::make_split(bias, 2, 1);
            m_input_map[DynamicQuantizeLSTMInput::B] = std::make_shared<ov::op::v1::Add>(split_bias.at(0), split_bias.at(1));
            m_input_map[DynamicQuantizeLSTMInput::B] =
                ov::op::util::convert_lstm_node_format(m_input_map[DynamicQuantizeLSTMInput::B],
                                                       ov::op::util::LSTMWeightsFormat::IOFC,
                                                       ov::op::util::LSTMWeightsFormat::FICO,
                                                       1);
        } else {
            auto b_shape = std::make_shared<ov::op::v0::Concat>(
                ov::OutputVector{num_directions_node,
                                 std::make_shared<ov::op::v1::Multiply>(
                                     ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {gates_count}),
                                     hidden_size_node)},
                0);
            m_input_map[DynamicQuantizeLSTMInput::B] = std::make_shared<ov::op::v3::Broadcast>(
                ov::op::v0::Constant::create(m_input_map[DynamicQuantizeLSTMInput::X].get_element_type(), ov::Shape{}, {0}),
                b_shape);
        }

        if (common::is_input_valid(node, 4)) {
            m_input_map[DynamicQuantizeLSTMInput::SEQUENCE_LENS] = ng_inputs.at(4);
        } else {
            m_input_map[DynamicQuantizeLSTMInput::SEQUENCE_LENS] =
                std::make_shared<ov::op::v3::Broadcast>(seq_length_node, batch_size_node);
        }

        if (common::is_input_valid(node, 5)) {
            auto init_h = normalize_tensor_rank(ng_inputs.at(5), 3, "initial_h");
            m_input_map[DynamicQuantizeLSTMInput::INITIAL_H] = ov::op::util::reorder_axes(init_h, {1, 0, 2});
        } else {
            auto init_h_shape = std::make_shared<ov::op::v0::Concat>(
                ov::OutputVector{batch_size_node, num_directions_node, hidden_size_node},
                0);
            m_input_map[DynamicQuantizeLSTMInput::INITIAL_H] = std::make_shared<ov::op::v3::Broadcast>(
                ov::op::v0::Constant::create(m_input_map[DynamicQuantizeLSTMInput::X].get_element_type(), ov::Shape{}, {0}),
                init_h_shape);
        }

        if (common::is_input_valid(node, 6)) {
            auto init_c = normalize_tensor_rank(ng_inputs.at(6), 3, "initial_c");
            m_input_map[DynamicQuantizeLSTMInput::INITIAL_C] = ov::op::util::reorder_axes(init_c, {1, 0, 2});
        } else {
            auto init_c_shape = std::make_shared<ov::op::v0::Concat>(
                ov::OutputVector{batch_size_node, num_directions_node, hidden_size_node},
                0);
            m_input_map[DynamicQuantizeLSTMInput::INITIAL_C] = std::make_shared<ov::op::v3::Broadcast>(
                ov::op::v0::Constant::create(m_input_map[DynamicQuantizeLSTMInput::X].get_element_type(), ov::Shape{}, {0}),
                init_c_shape);
        }
    }

    ov::Output<ov::Node>& at(const DynamicQuantizeLSTMInput& key) {
        return m_input_map.at(key);
    }

    std::map<DynamicQuantizeLSTMInput, ov::Output<ov::Node>> m_input_map;
};

struct LSTMAttributes {
    explicit LSTMAttributes(const ov::frontend::onnx::Node& node)
        : m_hidden_size{node.get_attribute_value<std::int64_t>("hidden_size")},
          m_clip_threshold{node.get_attribute_value<float>("clip", 0.f)},
          m_activations{node.get_attribute_value<std::vector<std::string>>("activations", {"sigmoid", "tanh", "tanh"})},
          m_activation_alpha{node.get_attribute_value<std::vector<float>>("activation_alpha", std::vector<float>{})},
          m_activation_beta{node.get_attribute_value<std::vector<float>>("activation_beta", std::vector<float>{})},
          m_input_forget{static_cast<bool>(node.get_attribute_value<std::int64_t>("input_forget", 0))} {
        m_clip_threshold = std::abs(m_clip_threshold);
        const auto direction = ov::util::to_lower(node.get_attribute_value<std::string>("direction", "forward"));
        m_direction = ov::as_enum<ov::op::RecurrentSequenceDirection>(direction);
    }

    ov::op::RecurrentSequenceDirection m_direction;
    std::int64_t m_hidden_size;
    float m_clip_threshold;
    std::vector<std::string> m_activations;
    std::vector<float> m_activation_alpha;
    std::vector<float> m_activation_beta;
    bool m_input_forget;
};

}  // namespace

namespace opset_1 {

ov::OutputVector dynamic_quantize_lstm(const ov::frontend::onnx::Node& node) {
    common::default_op_checks(node, 11, 12);

    LSTMAttributes attributes{node};
    DynamicQuantizeLSTMInputMap input_map{node, attributes.m_hidden_size};

    auto lstm_sequence = std::make_shared<ov::op::v5::LSTMSequence>(input_map.at(DynamicQuantizeLSTMInput::X),
                                                                    input_map.at(DynamicQuantizeLSTMInput::INITIAL_H),
                                                                    input_map.at(DynamicQuantizeLSTMInput::INITIAL_C),
                                                                    input_map.at(DynamicQuantizeLSTMInput::SEQUENCE_LENS),
                                                                    input_map.at(DynamicQuantizeLSTMInput::W),
                                                                    input_map.at(DynamicQuantizeLSTMInput::R),
                                                                    input_map.at(DynamicQuantizeLSTMInput::B),
                                                                    attributes.m_hidden_size,
                                                                    attributes.m_direction,
                                                                    attributes.m_activation_alpha,
                                                                    attributes.m_activation_beta,
                                                                    attributes.m_activations,
                                                                    attributes.m_clip_threshold);

    const auto Y = lstm_sequence->output(0);
    const auto Y_h = lstm_sequence->output(1);
    const auto Y_c = lstm_sequence->output(2);

    return {ov::op::util::reorder_axes(Y, {2, 1, 0, 3}),
            ov::op::util::reorder_axes(Y_h, {1, 0, 2}),
            ov::op::util::reorder_axes(Y_c, {1, 0, 2})};
}

ONNX_OP("DynamicQuantizeLSTM", OPSET_SINCE(1), com_microsoft::opset_1::dynamic_quantize_lstm, MICROSOFT_DOMAIN);

}  // namespace opset_1
}  // namespace com_microsoft
}  // namespace onnx
}  // namespace frontend
}  // namespace ov
