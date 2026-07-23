// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <cmath>
#include <cstdint>
#include <map>
#include <memory>
#include <numeric>
#include <string>
#include <vector>

#include "core/null_node.hpp"
#include "core/operator_set.hpp"
#include "exceptions.hpp"
#include "openvino/decompositions/low_precision_dequantize.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/lstm_sequence.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/op/util/rnn_cell_base.hpp"
#include "openvino/util/common_util.hpp"
#include "utils/common.hpp"
#include "utils/recurrent.hpp"
#include "utils/reshape.hpp"
#include "utils/split.hpp"

namespace ov {
namespace frontend {
namespace onnx {
namespace com_microsoft {
namespace {

using ov::frontend::onnx::recurrent::normalize_tensor_rank;

enum class DynamicQuantizeLSTMInput {
    X,
    W,
    R,
    B,
    SEQUENCE_LENS,
    INITIAL_H,
    INITIAL_C,
};

// Transpose quantized gate weights from the com.microsoft layout
// [num_directions, K, 4*hidden_size] to the standard ONNX LSTM layout
// [num_directions, 4*hidden_size, K] expected by the dequantization and
// gate-reordering steps below.
ov::Output<ov::Node> prepare_quantized_weights(const ov::frontend::onnx::Node& node,
                                               const ov::Output<ov::Node>& weights,
                                               const std::string& input_name) {
    const auto& rank = weights.get_partial_shape().rank();
    CHECK_VALID_NODE(node,
                     rank.is_static() && rank.get_length() == 3,
                     "DynamicQuantizeLSTM input '",
                     input_name,
                     "' must have static rank 3. Got rank: ",
                     rank);

    // The com.microsoft spec always provides weights in [num_directions, K, 4*hidden_size] layout.
    // Transpose unconditionally to the LSTM layout [num_directions, 4*hidden_size, K].
    // A shape-based heuristic (checking which axis equals 4*hidden_size) is incorrect
    // when K == 4*hidden_size, and would silently produce wrong results.
    return ov::op::util::reorder_axes(weights, {0, 2, 1});
}

// Append trailing size-1 dims to scale/zero_point so right-aligned autobroadcast
// aligns them to the correct axes of prepared weights [num_dir, 4h, K].
// For Constants the reshape is applied in-place (no graph node) so MarkDequantization
// still fires; for runtime tensors an Unsqueeze fallback is used instead.
ov::Output<ov::Node> align_dequant_param(const ov::frontend::onnx::Node& node,
                                         const ov::Output<ov::Node>& param,
                                         const std::string& input_name) {
    const auto& pshape = param.get_partial_shape();
    const auto& rank = pshape.rank();
    if (rank.is_dynamic()) {
        return param;
    }
    CHECK_VALID_NODE(node,
                     rank.get_length() == 1 || rank.get_length() == 2,
                     "DynamicQuantizeLSTM input '",
                     input_name,
                     "' must be rank-1 [num_directions] or rank-2 [num_directions, 4*hidden_size]. Got rank: ",
                     rank.get_length());

    const auto const_node = ov::as_type_ptr<ov::op::v0::Constant>(param.get_node_shared_ptr());
    if (const_node) {
        ov::Shape new_shape = pshape.to_shape();
        while (new_shape.size() < 3) {
            new_shape.push_back(1);
        }
        return std::make_shared<ov::op::v0::Constant>(*const_node, new_shape);
    }

    // Fallback for non-constant inputs: insert Unsqueeze nodes.
    const auto dims_to_add = static_cast<int64_t>(3) - rank.get_length();
    std::vector<int64_t> axes(dims_to_add);
    std::iota(axes.begin(), axes.end(), rank.get_length());
    auto axes_const = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{axes.size()}, axes);
    return std::make_shared<ov::op::v0::Unsqueeze>(param, axes_const);
}

ov::Output<ov::Node> dequantize_weights(const ov::frontend::onnx::Node& node,
                                        const ov::Output<ov::Node>& prepared_weights,
                                        const ov::Output<ov::Node>& scale,
                                        const ov::Output<ov::Node>& zero_point,
                                        const std::string& weights_name) {
    const auto aligned_scale = align_dequant_param(node, scale, weights_name + "_scale");
    const ov::Output<ov::Node> aligned_zp = (zero_point.get_node() != nullptr && !ov::op::util::is_null(zero_point))
                                                ? align_dequant_param(node, zero_point, weights_name + "_zero_point")
                                                : ov::Output<ov::Node>{};
    return ov::decomposition::low_precision_dequantize(prepared_weights, aligned_scale, aligned_zp);
}

struct DynamicQuantizeLSTMInputMap {
    explicit DynamicQuantizeLSTMInputMap(const ov::frontend::onnx::Node& node, int64_t hidden_size) {
        const auto& ng_inputs = node.get_ov_inputs();
        constexpr std::size_t gates_count{4};

        auto input_x = normalize_tensor_rank(ng_inputs.at(0), 3, "DynamicQuantizeLSTM", "X");
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
                         common::is_input_valid(node, 8),
                         "DynamicQuantizeLSTM mandatory input 'W_scale' (index 8) is missing.");
        CHECK_VALID_NODE(node,
                         ng_inputs.at(8).get_element_type() == ov::element::f32,
                         "DynamicQuantizeLSTM input 'W_scale' must have element type float32. Got: ",
                         ng_inputs.at(8).get_element_type());
        CHECK_VALID_NODE(node,
                         common::is_input_valid(node, 10),
                         "DynamicQuantizeLSTM mandatory input 'R_scale' (index 10) is missing.");
        CHECK_VALID_NODE(node,
                         ng_inputs.at(10).get_element_type() == ov::element::f32,
                         "DynamicQuantizeLSTM input 'R_scale' must have element type float32. Got: ",
                         ng_inputs.at(10).get_element_type());

        // Peephole weights (input P, index 7) are not supported: ov::op::v5::LSTMSequence has
        // no peephole inputs. Supporting P would require unrolling to per-step ov::op::v0::LSTMCell
        // calls (which does accept P). Reject rather than silently ignore.
        // TODO: support peephole input P by unrolling to LSTMCell.
        CHECK_VALID_NODE(node,
                         !common::is_input_valid(node, 7),
                         "DynamicQuantizeLSTM peephole input 'P' is not supported.");

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

        auto prepared_w = prepare_quantized_weights(node, ng_inputs.at(1), "W");
        auto prepared_r = prepare_quantized_weights(node, ng_inputs.at(2), "R");

        // Validate hidden_size attribute against R's actual shape.
        // R input has layout [num_dir, hidden_size, 4*hidden_size] (same [num_dir, K, 4*hidden_size]
        // convention used by prepare_quantized_weights, where K == hidden_size for R).
        // So dim 1 of the original R input equals hidden_size.
        const auto& r_pshape = ng_inputs.at(2).get_partial_shape();
        if (r_pshape.rank().is_static() && r_pshape[1].is_static()) {
            CHECK_VALID_NODE(node,
                             r_pshape[1].get_length() == hidden_size,
                             "DynamicQuantizeLSTM: 'hidden_size' attribute (",
                             hidden_size,
                             ") does not match R input shape dim 1 (",
                             r_pshape[1].get_length(),
                             ").");
        }

        m_input_map[DynamicQuantizeLSTMInput::W] = ov::op::util::convert_lstm_node_format(
            dequantize_weights(node, prepared_w, ng_inputs.at(8), w_zero_point, "W"),
            ov::op::util::LSTMWeightsFormat::IOFC,
            ov::op::util::LSTMWeightsFormat::FICO,
            1);
        m_input_map[DynamicQuantizeLSTMInput::R] = ov::op::util::convert_lstm_node_format(
            dequantize_weights(node, prepared_r, ng_inputs.at(10), r_zero_point, "R"),
            ov::op::util::LSTMWeightsFormat::IOFC,
            ov::op::util::LSTMWeightsFormat::FICO,
            1);

        // Data-bearing edges (X, W, R, provided initial states) are built explicitly above
        // so future activation-quantization insertion has a clear insertion point.
        const recurrent::LSTMDimensions dims{m_input_map[DynamicQuantizeLSTMInput::X],
                                             m_input_map[DynamicQuantizeLSTMInput::R]};
        const auto x_type = m_input_map[DynamicQuantizeLSTMInput::X].get_element_type();

        if (common::is_input_valid(node, 3)) {
            auto bias = normalize_tensor_rank(ng_inputs.at(3), 2, "DynamicQuantizeLSTM", "B");
            auto split_bias = ov::op::util::make_split(bias, 2, 1);
            m_input_map[DynamicQuantizeLSTMInput::B] =
                std::make_shared<ov::op::v1::Add>(split_bias.at(0), split_bias.at(1));
            m_input_map[DynamicQuantizeLSTMInput::B] =
                ov::op::util::convert_lstm_node_format(m_input_map[DynamicQuantizeLSTMInput::B],
                                                       ov::op::util::LSTMWeightsFormat::IOFC,
                                                       ov::op::util::LSTMWeightsFormat::FICO,
                                                       1);
        } else {
            m_input_map[DynamicQuantizeLSTMInput::B] = recurrent::default_bias(dims, x_type, gates_count);
        }

        if (common::is_input_valid(node, 4)) {
            m_input_map[DynamicQuantizeLSTMInput::SEQUENCE_LENS] = ng_inputs.at(4);
        } else {
            m_input_map[DynamicQuantizeLSTMInput::SEQUENCE_LENS] = recurrent::default_sequence_lens(dims);
        }

        if (common::is_input_valid(node, 5)) {
            auto init_h = normalize_tensor_rank(ng_inputs.at(5), 3, "DynamicQuantizeLSTM", "initial_h");
            m_input_map[DynamicQuantizeLSTMInput::INITIAL_H] = ov::op::util::reorder_axes(init_h, {1, 0, 2});
        } else {
            m_input_map[DynamicQuantizeLSTMInput::INITIAL_H] = recurrent::default_initial_state(dims, x_type);
        }

        if (common::is_input_valid(node, 6)) {
            auto init_c = normalize_tensor_rank(ng_inputs.at(6), 3, "DynamicQuantizeLSTM", "initial_c");
            m_input_map[DynamicQuantizeLSTMInput::INITIAL_C] = ov::op::util::reorder_axes(init_c, {1, 0, 2});
        } else {
            m_input_map[DynamicQuantizeLSTMInput::INITIAL_C] = recurrent::default_initial_state(dims, x_type);
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

    auto lstm_sequence =
        std::make_shared<ov::op::v5::LSTMSequence>(input_map.at(DynamicQuantizeLSTMInput::X),
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
