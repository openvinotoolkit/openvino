// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/split.hpp"

//#include <climits>

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/util/framework_node.hpp"
#include "openvino/op/variadic_split.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

using namespace ov::op;

OutputVector translate_chunk_fx(const NodeContext& context) {
    num_inputs_check(context, 3, 3);
    auto num_chunks = context.const_input<int>(1);
    auto dim = context.get_input(2);

    std::shared_ptr<ov::Node> chunk;
    auto dim_val = context.const_input<int>(2);
    auto shape = context.get_input(0).get_shape();
    if (dim_val < 0) {
        dim_val = static_cast<int>(shape.size()) + dim_val;
    }
    int num_splits = static_cast<int>(shape[dim_val]) / num_chunks;

    chunk = context.mark_node(std::make_shared<v1::Split>(context.get_input(0), dim, num_splits));

    return {context.mark_node(make_list_construct(chunk->outputs()))};
}

OutputVector translate_unbind_int_fx(const NodeContext& context) {
    num_inputs_check(context, 2, 3);
    auto input = context.get_input(0);
    auto dim = context.get_input(1);
    auto dim_val = context.const_input<int>(1);
    auto shape = input.get_shape();

    if (dim_val < 0) {
        dim_val = static_cast<int>(shape.size()) + dim_val;
    }

    auto num_splits = static_cast<int>(shape[dim_val]);
    auto chunk = context.mark_node(std::make_shared<v1::Split>(input, dim, num_splits));

    return {context.mark_node(make_list_construct(chunk->outputs()))};
}

OutputVector translate_split_with_sizes_fx(const NodeContext& context) {
    num_inputs_check(context, 3, 3);
    auto data = context.get_input(0);
    auto split_lengths = context.get_input(1);
    auto dim = context.get_input(2);

    auto split = std::make_shared<v1::VariadicSplit>(data, dim, split_lengths);

    return {context.mark_node(make_list_construct(split->outputs()))};
}

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
