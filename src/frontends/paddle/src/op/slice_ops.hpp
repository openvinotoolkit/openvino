// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include <limits.h>

#include "default_opset.hpp"
#include "openvino/frontend/paddle/node_context.hpp"

namespace ov {
namespace frontend {
namespace paddle {
namespace op {
namespace {
Output<Node> idx_node(const std::string& tensor_alias,
                      const std::string& list_alias,
                      const std::string& attr_alias,
                      const NodeContext& node) {
    if (node.has_input(tensor_alias)) {
        return std::make_shared<default_opset::Convert>(node.get_input(tensor_alias), element::i32);
    } else if (node.has_input(list_alias)) {
        auto inputs = node.get_ng_inputs(list_alias);
        return std::make_shared<default_opset::Convert>(std::make_shared<default_opset::Concat>(inputs, 0),
                                                        element::i32);
    } else {
        auto values = node.get_attribute<std::vector<int32_t>>(attr_alias);
        return default_opset::Constant::create(element::i32, {values.size()}, values);
    }
}
NamedOutputs slice_op(const NodeContext& node, const bool& stride_input) {
    const auto data = node.get_input("Input");
    const auto axes = node.get_attribute<std::vector<int32_t>>("axes");

    Output<Node> start_idx_node = idx_node("StartsTensor", "StartsTensorList", "starts", node);
    Output<Node> end_idx_node = idx_node("EndsTensor", "EndsTensorList", "ends", node);
    Output<Node> strides_idx_node;
    if (stride_input)
        strides_idx_node = idx_node("StridesTensor", "StridesTensorList", "strides", node);

    // The following process is:
    // Given:
    // data = [ [1, 2, 3, 4], [5, 6, 7, 8], ] // shape is: [2, 4]
    // axes = [0]
    // starts = [1]
    // ends = [2]
    // Our process is:
    //  1. Get 'axes': [0, 1], 'starts', 'ends'
    //  2. Get data shape: [2,4] and dims: 2
    //  3. Create two tensor t1 and t2, shape is the dims from step2: 2. t1: [0, 0], t2: [INT_MAX, INT_MAX]
    //  4. Use 'ScatterNDUpdate' to update some elements in t1, the updated indexes are coming from 'axes', the contents
    //  are coming from 'starts', t1: [1, 0]; apply the similar process to t2
    //  5. Call 'StrideSlice' with t1 and t2
    // Why using ScatterNDUpdate is that 'axes' may be discontinuous.

    // the shape of input, such as [2, 4]
    const auto shape_node = std::make_shared<default_opset::ShapeOf>(data, element::Type_t::i32);
    // the input dim, such as [2]
    const auto rank_node = std::make_shared<default_opset::ShapeOf>(shape_node, element::i32);
    const auto const_0_node = default_opset::Constant::create(element::i32, {}, {0});
    const auto const_max_node = default_opset::Constant::create(element::i32, {}, {INT_MAX});
    const auto const_1_node = default_opset::Constant::create(element::i32, {}, {1});
    // t1: [0, 0]
    const auto start_node = std::make_shared<default_opset::Broadcast>(const_0_node, rank_node);
    // t2: [INT_MAX, INT_MAX]
    const auto end_node = std::make_shared<default_opset::Broadcast>(const_max_node, rank_node);
    const auto strides_node = std::make_shared<default_opset::Broadcast>(const_1_node, rank_node);
    const auto axes_node = default_opset::Constant::create(element::i32, {axes.size(), 1}, axes);
    // update t1
    const auto fixed_start_node =
        std::make_shared<default_opset::ScatterNDUpdate>(start_node, axes_node, start_idx_node);
    // update t2
    const auto fixed_end_node = std::make_shared<default_opset::ScatterNDUpdate>(end_node, axes_node, end_idx_node);
    std::shared_ptr<Node> stride_slice_node;
    if (stride_input) {
        const auto fixed_strides_node =
            std::make_shared<default_opset::ScatterNDUpdate>(strides_node, axes_node, strides_idx_node);

        stride_slice_node = std::make_shared<default_opset::StridedSlice>(data,
                                                                          fixed_start_node,
                                                                          fixed_end_node,
                                                                          fixed_strides_node,
                                                                          std::vector<int64_t>{0},
                                                                          std::vector<int64_t>{0});
    } else {
        stride_slice_node = std::make_shared<default_opset::StridedSlice>(data,
                                                                          fixed_start_node,
                                                                          fixed_end_node,
                                                                          std::vector<int64_t>{0},
                                                                          std::vector<int64_t>{0});
    }

    const auto decrease_axis = node.get_attribute<std::vector<int32_t>>("decrease_axis");

    if (decrease_axis.size() > 0) {
        // according to paddle slice_op, when all axes are decreased, output shape is [1], instead of scalar.
        // Ref: paddle/fluid/operators/slice_op.h
        PartialShape input_shape = data.get_partial_shape();
        PADDLE_OP_CHECK(node,
                        input_shape.rank().is_static(),
                        "input rank of slice must be static when decrease_axis is set.");

        const auto squeeze_index_node =
            default_opset::Constant::create(element::i32, {decrease_axis.size()}, decrease_axis);
        const auto decreased_node = std::make_shared<default_opset::Squeeze>(stride_slice_node, squeeze_index_node);

        const auto input_rank = input_shape.rank().get_length();
        if ((size_t)input_rank == decrease_axis.size()) {
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
}  // namespace
}  // namespace op
}  // namespace paddle
}  // namespace frontend
}  // namespace ov
