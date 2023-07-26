// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/split.hpp"

#include <climits>

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/util/framework_node.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

using namespace ov::op;

OutputVector translate_chunk_fx(const NodeContext& context) {
    num_inputs_check(context, 2, 3);
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

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
