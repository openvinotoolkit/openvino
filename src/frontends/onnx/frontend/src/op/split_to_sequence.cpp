// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <cmath>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "core/operator_set.hpp"
#include "exceptions.hpp"
#include "openvino/core/node.hpp"
#include "openvino/core/validation_util.hpp"
#include "openvino/frontend/sequence_mark.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/split.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/op/util/framework_node.hpp"
#include "openvino/op/variadic_split.hpp"
#include "utils/common.hpp"

namespace {

/// @brief Gets and validates the 'axis' attribute from the node
/// @param node Input ONNX node
/// @param partial_shape Partial shape of the input tensor to be split
/// @return The validated axis value (may be non-negative even for dynamic-rank inputs)
std::int64_t get_axis(const ov::frontend::onnx::Node& node, const ov::PartialShape& partial_shape) {
    auto axis = node.get_attribute_value<std::int64_t>("axis", 0);

    const auto input_rank = partial_shape.rank();
    if (input_rank.is_static()) {
        if (axis < 0) {
            axis += input_rank.get_length();
        }
        OPENVINO_ASSERT(0 <= axis && axis < input_rank.get_length(), "SplitToSequence: axis is out of range");
    }
    // For dynamic rank: return axis as-is; assume caller will not use it for static range checks.

    return axis;
}

/// @brief Gets and validates the split axis length
/// @param partial_shape Partial shape of the input tensor to be split
/// @param axis The split axis
/// @return The validated axis length, or -1 if the dimension is dynamic
std::int64_t get_axis_length(const ov::PartialShape& partial_shape, std::int64_t axis) {
    if (!partial_shape.rank().is_static()) {
        return -1;
    }
    const auto axis_dimension = partial_shape[axis];
    if (!axis_dimension.is_static()) {
        return -1;
    }
    return axis_dimension.get_length();
}

/// @brief Implements the SplitToSequence operator with scalar 'split' input
/// @param node Input ONNX node
/// @param inputs Input tensors
/// @return A sequence of tensors obtained by splitting the input tensor in form of a SequenceMark node
ov::OutputVector split_with_scalar_split(const ov::frontend::onnx::Node& node, const ov::OutputVector& inputs) {
    const auto& input = inputs[0];
    const auto& partial_shape = input.get_partial_shape();
    const auto axis = get_axis(node, partial_shape);
    const auto& split = inputs[1];

    const auto split_const = ov::util::get_constant_from_source(split);
    OPENVINO_ASSERT(split_const != nullptr, "SplitToSequence: 'split' input must be a constant");
    const auto split_values = split_const->cast_vector<std::int64_t>();
    OPENVINO_ASSERT(!split_values.empty(), "SplitToSequence: 'split' input cannot be empty");

    const std::int64_t chunk = split_values.front();
    OPENVINO_ASSERT(chunk > 0, "SplitToSequence: scalar 'split' must be positive");

    const std::int64_t axis_length = get_axis_length(partial_shape, axis);
    if (axis_length < 0) {
        // Dynamic axis dimension: cannot enumerate outputs statically.
        // Return an empty sequence; the caller will create a stacked SequenceMark.
        return {};
    }

    std::vector<std::int64_t> lengths;
    lengths.reserve(static_cast<std::size_t>((axis_length + chunk - 1) / chunk));

    for (std::int64_t offset = 0; offset < axis_length; offset += chunk) {
        lengths.push_back(std::min(chunk, axis_length - offset));
    }

    const auto axis_const = ov::op::v0::Constant::create(ov::element::i64, {}, {axis});
    const auto split_lengths = ov::op::v0::Constant::create(ov::element::i64, {lengths.size()}, lengths);

    return std::make_shared<ov::op::v1::VariadicSplit>(input, axis_const, split_lengths)->outputs();
}

/// @brief Implements the SplitToSequence operator with 1D 'split' input
/// @param node Input ONNX node
/// @param inputs Input tensors
/// @return A sequence of tensors obtained by splitting the input tensor in form of a SequenceMark node
ov::OutputVector split_with_1d_split(const ov::frontend::onnx::Node& node, const ov::OutputVector& inputs) {
    const auto& input = inputs[0];
    const auto& split = inputs[1];
    const auto axis = node.get_attribute_as_constant<std::int64_t>("axis", 0);

    return std::make_shared<ov::op::v1::VariadicSplit>(input, axis, split)->outputs();
}

/// @brief Implements the SplitToSequence operator with explicit 'split' input
/// @param node Input ONNX node
/// @return A sequence of tensors obtained by splitting the input tensor in form of a SequenceMark node
ov::OutputVector split_with_explicit_split(const ov::frontend::onnx::Node& node) {
    const auto& inputs = node.get_ov_inputs();

    const auto& split = inputs[1];

    const auto split_rank = split.get_partial_shape().rank();

    if (split_rank.is_dynamic()) {
        // Cannot determine scalar vs 1-D at compile time; treat as 1-D (most common case).
        return split_with_1d_split(node, inputs);
    }

    const auto rank = split_rank.get_length();

    if (0 == rank) {
        return split_with_scalar_split(node, inputs);
    } else if (1 == rank) {
        return split_with_1d_split(node, inputs);
    } else {
        OPENVINO_THROW("SplitToSequence: 'split' input must be a scalar or 1-D tensor");
    }
}

/// @brief Implements the SplitToSequence operator with default 'split' input
/// @param node Input ONNX node
/// @return A sequence of tensors obtained by splitting the input tensor in form of a SequenceMark node
ov::OutputVector split_with_default_split(const ov::frontend::onnx::Node& node) {
    const auto& inputs = node.get_ov_inputs();
    const auto& input = inputs[0];
    const auto& partial_shape = input.get_partial_shape();
    const auto axis = get_axis(node, partial_shape);

    const std::int64_t axis_length = get_axis_length(partial_shape, axis);

    if (axis_length < 0) {
        // Dynamic axis dimension: cannot enumerate outputs statically.
        // Return an empty sequence; the caller will wrap as a stacked SequenceMark.
        return {};
    }

    ov::OutputVector output_sequence;

    if (0 == axis_length) {
        return output_sequence;
    }

    output_sequence.reserve(axis_length);

    const auto axis_const = ov::op::v0::Constant::create(ov::element::i64, {}, {axis});

    auto split_op = std::make_shared<ov::op::v1::Split>(input, axis_const, static_cast<std::size_t>(axis_length));

    const auto keepdims = node.get_attribute_value<std::int64_t>("keepdims", 1) == 1;

    const auto squeeze_axes = ov::op::v0::Constant::create(ov::element::i64, {1}, {axis});

    for (std::int64_t offset = 0; offset < axis_length; ++offset) {
        auto element = split_op->output(static_cast<std::size_t>(offset));

        if (!keepdims) {
            element = std::make_shared<ov::op::v15::Squeeze>(element, squeeze_axes);
        }
        output_sequence.push_back(std::move(element));
    }

    return output_sequence;
}

}  // namespace

namespace ov {
namespace frontend {
namespace onnx {
namespace ai_onnx {
namespace opset_11 {

/// @brief Implements the SplitToSequence operator
/// @param node Input ONNX node
/// @return A sequence of tensors obtained by splitting the input tensor in form of a SequenceMark node.
///         When the split produces a statically-enumerable element list the SequenceMark holds those
///         elements directly.  When the number of elements cannot be determined at graph-construction
///         time (e.g. because the split-lengths tensor has a dynamic shape), the SequenceMark wraps
///         the *full input tensor* as a single element and carries "stacked=1" / "stacked_axis=N"
///         attributes so that SequenceAt can use Gather at runtime.
ov::OutputVector split_to_sequence(const ov::frontend::onnx::Node& node) {
    constexpr auto input_only = 1;
    constexpr auto input_and_split = 2;

    common::default_op_checks(node, input_only, input_and_split);

    constexpr auto split_input_index = 1;

    const auto& inputs = node.get_ov_inputs();
    const auto& input = inputs[0];
    const auto& partial_shape = input.get_partial_shape();

    // Determine upfront whether a stacked (dynamic) representation is required.
    // This avoids confusing a genuinely empty output_sequence (axis_length == 0) with the
    // dynamic case where split_with_* returns {} because output count is unknown.
    const auto axis_raw = node.get_attribute_value<std::int64_t>("axis", 0);
    bool needs_stacked = false;

    if (common::is_input_valid(node, split_input_index)) {
        const auto& split = inputs[1];
        const auto split_rank = split.get_partial_shape().rank();
        if (split_rank.is_dynamic()) {
            needs_stacked = true;
        } else if (split_rank.get_length() == 1) {
            // 1-D split: output count equals the split-lengths dimension.
            const auto& split_dim = split.get_partial_shape()[0];
            if (split_dim.is_dynamic()) {
                needs_stacked = true;
            }
        }
        // Rank-0 (scalar) split: output count = ceil(axis_dim / chunk).
        // needs_stacked stays false; split_with_scalar_split returns {} when axis_dim is dynamic.
        if (!needs_stacked && split_rank.is_static() && split_rank.get_length() == 0) {
            // axis dimension dynamic?
            if (partial_shape.rank().is_static()) {
                auto ax = axis_raw;
                if (ax < 0)
                    ax += partial_shape.rank().get_length();
                if (ax >= 0 && ax < partial_shape.rank().get_length() && !partial_shape[ax].is_static()) {
                    needs_stacked = true;
                }
            } else {
                needs_stacked = true;
            }
        }
    } else {
        // No explicit split: output count = axis dimension length.
        if (partial_shape.rank().is_dynamic()) {
            needs_stacked = true;
        } else {
            auto ax = axis_raw;
            if (ax < 0)
                ax += partial_shape.rank().get_length();
            if (ax >= 0 && ax < partial_shape.rank().get_length() && !partial_shape[ax].is_static()) {
                needs_stacked = true;
            }
        }
    }

    if (needs_stacked) {
        // Dynamic path: wrap the full input tensor and annotate for Gather-based access.
        std::int64_t stacked_axis = axis_raw;
        if (partial_shape.rank().is_static() && axis_raw < 0) {
            stacked_axis = axis_raw + partial_shape.rank().get_length();
        }
        auto seq_mark = std::make_shared<ov::frontend::SequenceMark>(ov::OutputVector{input});
        ov::op::util::FrameworkNodeAttrs attrs;
        attrs.set_type_name("SplitToSequenceMark");
        attrs["stacked"] = "1";
        attrs["stacked_axis"] = std::to_string(stacked_axis);
        seq_mark->set_attrs(attrs);
        return {seq_mark};
    }

    // Static path: enumerate elements.
    const auto output_sequence = common::is_input_valid(node, split_input_index) ? split_with_explicit_split(node)
                                                                                 : split_with_default_split(node);
    return {std::make_shared<ov::frontend::SequenceMark>(output_sequence)};
}

/// @brief Registers the SplitToSequence operator implementation in the ONNX frontend
/// @remark The operator is available since ONNX opset 11.
///         Registering as available since opset 1 for compatibility with existing tests.
ONNX_OP("SplitToSequence", OPSET_SINCE(1), ai_onnx::opset_11::split_to_sequence);

}  // namespace opset_11

}  // namespace ai_onnx
}  // namespace onnx
}  // namespace frontend
}  // namespace ov
