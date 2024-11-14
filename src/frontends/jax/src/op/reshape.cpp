// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/reshape.hpp"

#include <cstdint>
#include <vector>

#include "openvino/core/node.hpp"
#include "openvino/core/node_output.hpp"
#include "openvino/frontend/jax/node_context.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/transpose.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace jax {
namespace op {

using namespace ov::op;

OutputVector translate_reshape(const NodeContext& context) {
    num_inputs_check(context, 1, 1);
    Output<Node> input = context.get_input(0);
    auto new_sizes = context.const_named_param<std::vector<int64_t>>("new_sizes");
    if (context.has_param("dimensions")) {
        auto dimensions = context.const_named_param<std::vector<int64_t>>("dimensions");
        // transpose the input first.
        auto permutation_node = std::make_shared<v0::Constant>(element::i64, Shape{dimensions.size()}, dimensions);
        input = std::make_shared<v1::Transpose>(input, permutation_node);
    }

    auto new_shape_node = std::make_shared<v0::Constant>(element::i64, Shape{new_sizes.size()}, new_sizes);
    Output<Node> res = std::make_shared<v1::Reshape>(input, new_shape_node, false);
    return {res};
};

}  // namespace op
}  // namespace jax
}  // namespace frontend
}  // namespace ov