// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include <limits.h>

#include "default_opset.hpp"
#include "openvino/frontend/paddle/node_context.hpp"

namespace ov {
namespace frontend {
namespace paddle {
namespace op {
NamedOutputs strided_slice(const NodeContext& node) {
    const auto data = node.get_input("Input");
    const auto axes = node.get_attribute<std::vector<int32_t>>("axes");
    Output<Node> start_idx_node, end_idx_node, strides_idx_node;
    if (node.has_input("StartsTensor")) {
        start_idx_node = node.get_input("StartsTensor");
    } else if (node.has_input("StartsTensorList")) {
        auto inputs = node.get_ng_inputs("StartsTensorList");
        start_idx_node = std::make_shared<default_opset::Concat>(inputs, 0);
    } else {
        auto starts = node.get_attribute<std::vector<int32_t>>("starts");
        start_idx_node = default_opset::Constant::create(element::i32, {starts.size()}, starts);
    }

    if (node.has_input("EndsTensor")) {
        end_idx_node = node.get_input("EndsTensor");
    } else if (node.has_input("EndsTensorList")) {
        auto inputs = node.get_ng_inputs("EndsTensorList");
        end_idx_node = std::make_shared<default_opset::Concat>(inputs, 0);
    } else {
        auto ends = node.get_attribute<std::vector<int32_t>>("ends");
        end_idx_node = default_opset::Constant::create(element::i32, {ends.size()}, ends);
    }

    if (node.has_input("StridesTensor")) {
        strides_idx_node = node.get_input("StridesTensor");
    } else if (node.has_input("StridesTensorList")) {
        auto inputs = node.get_ng_inputs("StridesTensorList");
        strides_idx_node = std::make_shared<default_opset::Concat>(inputs, 0);
    } else {
        auto strides = node.get_attribute<std::vector<int32_t>>("strides");
        strides_idx_node = default_opset::Constant::create(element::i32, {strides.size()}, strides);
    }

    const auto shape_node = std::make_shared<default_opset::ShapeOf>(data, element::Type_t::i32);
    const auto shape_shape_node = std::make_shared<default_opset::ShapeOf>(shape_node, element::i32);
    const auto const_0_node = default_opset::Constant::create(element::i32, {}, {0});
    const auto const_max_node = default_opset::Constant::create(element::i32, {}, {INT_MAX});
    const auto const_1_node = default_opset::Constant::create(element::i32, {}, {1});
    const auto start_node = std::make_shared<default_opset::Broadcast>(const_0_node, shape_shape_node);
    const auto end_node = std::make_shared<default_opset::Broadcast>(const_max_node, shape_shape_node);
    const auto strides_node = std::make_shared<default_opset::Broadcast>(const_1_node, shape_shape_node);
    const auto axes_node = default_opset::Constant::create(element::i32, {axes.size(), 1}, axes);
    const auto fixed_start_node =
        std::make_shared<default_opset::ScatterNDUpdate>(start_node, axes_node, start_idx_node);
    const auto fixed_end_node = std::make_shared<default_opset::ScatterNDUpdate>(end_node, axes_node, end_idx_node);
    const auto fixed_strides_node =
        std::make_shared<default_opset::ScatterNDUpdate>(strides_node, axes_node, strides_idx_node);

    const auto stride_slice_node = std::make_shared<default_opset::StridedSlice>(data,
                                                                                 fixed_start_node,
                                                                                 fixed_end_node,
                                                                                 fixed_strides_node,
                                                                                 std::vector<int64_t>{0},
                                                                                 std::vector<int64_t>{0});

    const auto decrease_axis = node.get_attribute<std::vector<int32_t>>("decrease_axis");

    if (decrease_axis.size() > 0) {
        PartialShape input_shape = data.get_partial_shape();
        PADDLE_OP_CHECK(node,
                        input_shape.rank().is_static(),
                        "input rank of slice must be static when decrease_axis is set.");

        const auto squeeze_index_node =
            default_opset::Constant::create(element::i32, {decrease_axis.size()}, decrease_axis);
        const auto decreased_node = std::make_shared<default_opset::Squeeze>(stride_slice_node, squeeze_index_node);

        const auto input_rank = input_shape.rank().get_length();
        if (input_rank == decrease_axis.size()) {
            auto restore_node = std::make_shared<default_opset::Reshape>(
                decreased_node,
                std::make_shared<default_opset::Constant>(element::i64, Shape{1}, 1),
                false);  // restore to shape (1,)
            return node.default_single_output_mapping({restore_node}, {"Out"});
        }

        return node.default_single_output_mapping({decreased_node}, {"Out"});
    }

    return node.default_single_output_mapping({stride_slice_node}, {"Out"});
}
}  // namespace op
}  // namespace paddle
}  // namespace frontend
}  // namespace ov