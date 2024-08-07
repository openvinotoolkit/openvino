// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/squeeze.hpp"

#include "openvino/frontend/jax/node_context.hpp"
#include "openvino/op/constant.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace jax {
namespace op {

using namespace ov::op;

OutputVector translate_squeeze(const NodeContext& context) {
    num_inputs_check(context, 1, 1);
    auto x = context.get_input(0);
    auto dimensions = context.const_named_param<std::vector<int64_t>>("dimensions");
    auto dimensions_node = std::make_shared<v0::Constant>(element::i64, Shape{dimensions.size()}, dimensions);
    return {std::make_shared<v0::Squeeze>(x, dimensions_node)};
};

}  // namespace op
}  // namespace jax
}  // namespace frontend
}  // namespace ov
