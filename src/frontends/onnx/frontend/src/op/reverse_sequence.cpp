// frontends/onnx/frontend/src/op/reverse_sequence.cpp
// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/reverse_sequence.hpp"

#include "core/operator_set.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/frontend/exception.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/equal.hpp"
#include "openvino/op/select.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "utils/common.hpp"

using namespace ov::op;

namespace ov {
namespace frontend {
namespace onnx {
namespace ai_onnx {
namespace opset_1 {

ov::OutputVector reverse_sequence(const ov::frontend::onnx::Node& node) {
    const auto data = node.get_ov_inputs().at(0);
    const auto sequence_lengths = node.get_ov_inputs().at(1);

    // Convert sequence_lengths to int32 (internal normalization)
    const auto sequence_lengths_i32 = std::make_shared<v0::Convert>(sequence_lengths, ov::element::i32);
    const auto data_rank = data.get_partial_shape().rank();

    const auto batch_axis = node.get_attribute_value<int64_t>("batch_axis", 1);
    const auto normalized_batch_axis = common::normalize_axis(node.get_description(), batch_axis, data_rank);
    const auto time_axis = node.get_attribute_value<int64_t>("time_axis", 0);
    const auto normalized_time_axis = common::normalize_axis(node.get_description(), time_axis, data_rank);

    FRONT_END_GENERAL_CHECK(normalized_batch_axis == 0 || normalized_batch_axis == 1,
                            "Allowed values of 'batch_axis' are 0 or 1.");
    FRONT_END_GENERAL_CHECK(normalized_time_axis == 0 || normalized_time_axis == 1,
                            "Allowed values of 'time_axis' are 0 or 1.");
    FRONT_END_GENERAL_CHECK(normalized_batch_axis != normalized_time_axis,
                            "'batch_axis' and 'time_axis' cannot be the same.");

    // Base ReverseSequence op
    auto rev_node =
        std::make_shared<v0::ReverseSequence>(data, sequence_lengths_i32, normalized_batch_axis, normalized_time_axis);

    // === Handle case where sequence_lengths == 0 ===

    FRONT_END_GENERAL_CHECK(data_rank.is_static(), "ReverseSequence translation requires static input rank.");

    const int64_t rank_len = data_rank.get_length();

    // Step 1: sequence_lengths == 0 → [B]
    auto seq_len_f = std::make_shared<v0::Convert>(sequence_lengths_i32, ov::element::f32);
    auto zero_f = v0::Constant::create(ov::element::f32, ov::Shape{}, {0.0f});
    auto eq_mask = std::make_shared<v1::Equal>(seq_len_f, zero_f);  // [B]

    // Step 2: Expand mask correctly based on batch_axis
    ov::Output<ov::Node> expanded_mask = eq_mask;
    if (normalized_batch_axis == 0) {
        // batch along axis 0 → unsqueeze remaining axes [1..rank_len-1]
        for (int64_t axis = 1; axis < rank_len; ++axis) {
            auto axis_const = v0::Constant::create(ov::element::i64, {1}, {axis});
            expanded_mask = std::make_shared<v0::Unsqueeze>(expanded_mask, axis_const);
        }
    } else if (normalized_batch_axis == 1) {
        // batch along axis 1 → unsqueeze axis 0 first, then [2..rank_len-1]
        auto axis0 = v0::Constant::create(ov::element::i64, {1}, {0});
        expanded_mask = std::make_shared<v0::Unsqueeze>(expanded_mask, axis0);
        for (int64_t axis = 2; axis < rank_len; ++axis) {
            auto axis_const = v0::Constant::create(ov::element::i64, {1}, {axis});
            expanded_mask = std::make_shared<v0::Unsqueeze>(expanded_mask, axis_const);
        }
    }

    // Step 3: Broadcast mask to full data shape
    auto shape_of_data = std::make_shared<v3::ShapeOf>(data);
    auto mask_broadcast = std::make_shared<v3::Broadcast>(expanded_mask, shape_of_data->output(0));

    // Step 4: Broadcast zeros to data shape
    auto zero_scalar = v0::Constant::create(data.get_element_type(), ov::Shape{}, {0.0f});
    auto zero_expanded = std::make_shared<v3::Broadcast>(zero_scalar, shape_of_data->output(0));

    // Step 5: Select(mask → zero, else → reversed)
    auto final_output = std::make_shared<v1::Select>(mask_broadcast, zero_expanded, rev_node);

    return {final_output};
}

ONNX_OP("ReverseSequence", OPSET_SINCE(1), ai_onnx::opset_1::reverse_sequence);

}  // namespace opset_1
}  // namespace ai_onnx
}  // namespace onnx
}  // namespace frontend
}  // namespace ov
