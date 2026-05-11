// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <cstdint>
#include <memory>

#include "core/operator_set.hpp"
#include "exceptions.hpp"
#include "openvino/core/type.hpp"
#include "openvino/core/validation_util.hpp"
#include "openvino/frontend/exception.hpp"
#include "openvino/frontend/sequence_insert.hpp"
#include "openvino/frontend/sequence_mark.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/less.hpp"
#include "openvino/op/select.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "utils/common.hpp"

namespace ov {
namespace frontend {
namespace onnx {
namespace ai_onnx {
namespace opset_11 {

namespace {

/// @brief Normalises a dynamic scalar position and Gathers the element from a stacked tensor.
/// @param stacked The stacked tensor (elements along @p axis).
/// @param position Scalar i64 position (may be dynamic; negatives are normalised with ShapeOf).
/// @param axis Axis along which elements are stacked.
/// @return The gathered element output.
ov::Output<ov::Node> gather_at_dynamic_position(const ov::Output<ov::Node>& stacked,
                                                const ov::Output<ov::Node>& position,
                                                std::int64_t axis) {
    namespace op = ov::op;
    ov::Output<ov::Node> pos_i64 = std::make_shared<op::v0::Convert>(position, ov::element::i64)->output(0);
    const auto axis_const = op::v0::Constant::create(ov::element::i64, {}, {axis});

    const auto& stacked_shape = stacked.get_partial_shape();
    const bool axis_dim_static = stacked_shape.rank().is_static() && axis < stacked_shape.rank().get_length() &&
                                 stacked_shape[static_cast<std::ptrdiff_t>(axis)].is_static();

    if (axis_dim_static) {
        // Normalise negative index at compile time if axis length is known
        const auto seq_len = stacked_shape[static_cast<std::ptrdiff_t>(axis)].get_length();
        const auto seq_len_const = op::v0::Constant::create(ov::element::i64, {}, {seq_len});
        const auto zero = op::v0::Constant::create(ov::element::i64, {}, {0LL});
        auto is_neg = std::make_shared<op::v1::Less>(pos_i64, zero);
        auto pos_adjusted = std::make_shared<op::v1::Add>(pos_i64, seq_len_const);
        pos_i64 = std::make_shared<op::v1::Select>(is_neg, pos_adjusted, pos_i64)->output(0);
    }
    // For dynamic axis length, Gather will apply modular semantics at runtime.

    return std::make_shared<op::v8::Gather>(stacked, pos_i64, axis_const);
}

}  // namespace

/// @brief Implements the SequenceAt operator
/// @param node Input ONNX node. Must have two inputs: a sequence and a position.
///             Sequence is represented as:
///               - a SequenceMark node (from SequenceEmpty/SplitToSequence/SequenceConstruct), or
///               - a SequenceInsert chain (append-only, no positional arguments), or
///               - a plain tensor (outer-scope sequence passed through a Loop boundary).
/// @return The tensor at the specified position in the input sequence
ov::OutputVector sequence_at(const ov::frontend::onnx::Node& node) {
    constexpr auto input_sequence_and_position = 2;

    common::default_op_checks(node, input_sequence_and_position, input_sequence_and_position);

    const auto& inputs = node.get_ov_inputs();
    auto position = inputs[1];
    OPENVINO_ASSERT(position.get_partial_shape().rank().compatible(0), "SequenceAt: 'position' input must be a scalar");

    const auto& input_sequence = as_type_ptr<SequenceMark>(inputs[0].get_node_shared_ptr());

    // ── Path A: SequenceMark input ──────────────────────────────────────────────
    if (input_sequence) {
        const auto& sm_attrs = input_sequence->get_attrs();
        const bool is_stacked = sm_attrs.find("stacked") != sm_attrs.end();

        if (is_stacked) {
            // Dynamic-length sequence: the SequenceMark wraps the full input tensor.
            // SequenceAt(seq, pos) == Gather(inner_tensor, pos, stacked_axis).
            OPENVINO_ASSERT(input_sequence->get_input_size() == 1,
                            "SequenceAt: stacked SequenceMark must have exactly one input tensor");
            const auto stacked = input_sequence->input_value(0);
            const std::int64_t stacked_axis = std::stoi(sm_attrs.at("stacked_axis"));
            return {gather_at_dynamic_position(stacked, position, stacked_axis)};
        }

        // Static sequence: elements are listed directly in the SequenceMark.
        const auto position_const = ov::util::get_constant_from_source(position);
        if (position_const) {
            // Fast path: constant position.
            const auto position_value = position_const->cast_vector<std::int64_t>()[0];
            const auto seq_len = static_cast<std::int64_t>(input_sequence->get_sequence().size());
            const auto pos_norm = position_value < 0 ? position_value + seq_len : position_value;
            OPENVINO_ASSERT(pos_norm >= 0 && pos_norm < seq_len, "SequenceAt: 'position' is out of bounds");
            return {input_sequence->get_sequence().at(static_cast<std::size_t>(pos_norm))};
        }

        // Dynamic position on a static (known-element) sequence:
        // stack all elements along a new axis 0 and use Gather.
        const auto elements = input_sequence->get_sequence();
        OPENVINO_ASSERT(!elements.empty(), "SequenceAt: cannot handle dynamic position on empty sequence");

        const auto stack_axis_const = ov::op::v0::Constant::create(ov::element::i64, {1}, {0LL});
        ov::OutputVector unsqueezed;
        unsqueezed.reserve(elements.size());
        for (const auto& elem : elements) {
            unsqueezed.push_back(std::make_shared<ov::op::v0::Unsqueeze>(elem, stack_axis_const));
        }
        const auto stacked = std::make_shared<ov::op::v0::Concat>(unsqueezed, 0);
        return {gather_at_dynamic_position(stacked, position, 0)};
    }

    // ── Path A+: SequenceInsert chain (ONNX append-only pattern) ────────────────
    // Handles ONNX models using: SequenceEmpty → SequenceInsert × N → SequenceAt.
    // SequenceInsert produces a helper FrameworkNode (not a SequenceMark), so this
    // path walks the chain backward to collect the appended tensors, then indexes
    // into the resulting list.  Only supported when no SequenceInsert in the chain
    // carries a positional argument (i.e., all inserts are pure appends).
    if (const auto seq_insert_root = as_type_ptr<SequenceInsert>(inputs[0].get_node_shared_ptr())) {
        ov::OutputVector sequence_elements;
        ov::Output<ov::Node> current = inputs[0];
        bool all_append = true;

        while (const auto si = as_type_ptr<SequenceInsert>(current.get_node_shared_ptr())) {
            if (si->has_position()) {
                all_append = false;
                break;
            }
            sequence_elements.push_back(si->get_tensor());
            current = si->get_input_sequence();
        }

        if (all_append) {
            // Elements were collected newest-first; reverse to oldest-first order.
            std::reverse(sequence_elements.begin(), sequence_elements.end());

            const auto position_const = ov::util::get_constant_from_source(position);
            if (position_const) {
                const auto position_value = position_const->cast_vector<std::int64_t>()[0];
                const auto seq_len = static_cast<std::int64_t>(sequence_elements.size());
                const auto pos_norm = position_value < 0 ? position_value + seq_len : position_value;
                OPENVINO_ASSERT(pos_norm >= 0 && pos_norm < seq_len, "SequenceAt: 'position' is out of bounds");
                return {sequence_elements.at(static_cast<std::size_t>(pos_norm))};
            }

            // Dynamic position on a SequenceInsert chain: stack elements and Gather.
            OPENVINO_ASSERT(!sequence_elements.empty(),
                            "SequenceAt: cannot handle dynamic position on empty SequenceInsert chain");
            const auto stack_axis_const = ov::op::v0::Constant::create(ov::element::i64, {1}, {0LL});
            ov::OutputVector unsqueezed;
            unsqueezed.reserve(sequence_elements.size());
            for (const auto& elem : sequence_elements) {
                unsqueezed.push_back(std::make_shared<ov::op::v0::Unsqueeze>(elem, stack_axis_const));
            }
            const auto stacked = std::make_shared<ov::op::v0::Concat>(unsqueezed, 0);
            return {gather_at_dynamic_position(stacked, position, 0)};
        }
        // If not all-append, fall through to Path B for Loop-boundary handling.
    }

    // ── Path B: plain tensor input (outer-scope sequence through Loop boundary) ──
    // The sequence was passed to the Loop body as a Parameter (stacked tensor).
    // Default gather axis is 0 — consistent with the stacking done by split_to_sequence
    // when producing a stacked SequenceMark and with the materialisation in loop.cpp.
    return {gather_at_dynamic_position(inputs[0], position, 0)};
}

/// @brief Registers the SequenceAt operator implementation in the ONNX frontend
/// @remark The operator is available since ONNX opset 11.
///         Registering as available since opset 1 for compatibility with existing tests.
ONNX_OP("SequenceAt", OPSET_SINCE(1), ai_onnx::opset_11::sequence_at);

}  // namespace opset_11

}  // namespace ai_onnx
}  // namespace onnx
}  // namespace frontend
}  // namespace ov
