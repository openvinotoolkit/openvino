// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/opsets/opset10.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

OutputVector translate_chunk(NodeContext& context) {
    const auto input_tensor = context.get_input(0);
    int64_t dim, chunks;

    chunks = context.const_input<int64_t>(1);
    dim = context.const_input<int64_t>(2);

    auto zero_constant = context.mark_node(opset10::Constant::create(element::i32, Shape{1}, {0}));
    auto dim_constant = context.mark_node(opset10::Constant::create(element::i32, Shape{1}, {dim}));
    auto chunks_constant = context.mark_node(opset10::Constant::create(element::i32, Shape{1}, {chunks}));

    auto shape = context.mark_node(std::make_shared<opset10::ShapeOf>(input_tensor));
    auto dimension = context.mark_node(std::make_shared<opset10::Gather>(shape, dim_constant, zero_constant));
    auto input_size = context.mark_node(std::make_shared<opset10::Squeeze>(dimension));

    auto chunk_size_div = context.mark_node(std::make_shared<opset10::Divide>(input_size, chunks_constant, true));
    auto chunk_size = context.mark_node(std::make_shared<opset10::Floor>(chunk_size_div));
    auto last_chunk_size = context.mark_node(std::make_shared<opset10::Mod>(input_size, chunks_constant));
    auto nonzero_last_chunk = context.mark_node(std::make_shared<opset10::Greater>(last_chunk_size, zero_constant));

    auto computed_chunk_size = context.mark_node(std::make_shared<opset10::Add>(chunk_size, nonzero_last_chunk));
    auto computed_last_chunk_size = context.mark_node(std::make_shared<opset10::Mod>(input_size, computed_chunk_size));
    auto computed_last_chunk_idx =
        context.mark_node(std::make_shared<opset10::Subtract>(input_size, computed_last_chunk_size));

    auto slice_chunks = context.mark_node(std::make_shared<opset10::Slice>(input_tensor,
                                                                           zero_constant,
                                                                           computed_last_chunk_idx,
                                                                           computed_chunk_size,
                                                                           dim_constant));
    auto last_chunk = context.mark_node(std::make_shared<opset10::Slice>(input_tensor,
                                                                         computed_last_chunk_idx,
                                                                         input_size,
                                                                         computed_last_chunk_size,
                                                                         dim_constant));

    auto concatenated_chunks =
        context.mark_node(std::make_shared<opset10::Concat>(OutputVector{slice_chunks, last_chunk}, dim));

    return {concatenated_chunks};
};

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
