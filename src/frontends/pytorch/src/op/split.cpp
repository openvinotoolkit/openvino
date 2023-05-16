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

std::shared_ptr<ov::Node> translate_chunk_common(const NodeContext& context, const bool split_by_chunks) {
    num_inputs_check(context, 2, 3);
    auto num_chunks = context.const_input<int>(1);
    auto dim = context.get_input(2);

    std::shared_ptr<ov::Node> chunk;
    if (split_by_chunks) {
        chunk = context.mark_node(std::make_shared<v1::Split>(context.get_input(0), dim, num_chunks));
    } else {
        auto dim_val = context.const_input<int>(2);
        if (dim_val < 0) {
            dim_val = context.get_input(0).get_shape().size()+dim_val;
        }
        int num_splits = context.get_input(0).get_shape()[dim_val] / num_chunks;

        chunk = context.mark_node(std::make_shared<v1::Split>(context.get_input(0), dim, num_splits));
    }

    return chunk;
}

OutputVector translate_chunk(const NodeContext& context) {
    // Schema: aten::chunk(Tensor input, int chunks, int dim=0) -> Tensor
    auto chunk = translate_chunk_common(context, true);
    return {context.mark_output(chunk)};
};

OutputVector translate_chunk_fx(const NodeContext& context) {
    auto chunk = translate_chunk_common(context, false);
    return {context.mark_node(make_list_construct(chunk->outputs()))};
}

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
