// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <cstdint>
#include <vector>

#include "openvino/frontend/jax/node_context.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/constant.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace jax {
namespace op {

using namespace ov::op;

OutputVector translate_broadcast_in_dim(const NodeContext& context) {
    num_inputs_check(context, 1, 1);
    auto x = context.get_input(0);
    auto shape_vector = context.const_named_param<std::vector<int64_t>>("shape");
    auto broadcast_dimensions_vector = context.const_named_param<std::vector<int64_t>>("broadcast_dimensions");

    auto shape = std::make_shared<v0::Constant>(element::i64, Shape{shape_vector.size()}, shape_vector);
    auto broadcast_dimensions = std::make_shared<v0::Constant>(element::i64,
                                                               Shape{broadcast_dimensions_vector.size()},
                                                               broadcast_dimensions_vector);
    return {std::make_shared<v3::Broadcast>(x, shape, broadcast_dimensions, BroadcastType::EXPLICIT)};
};

}  // namespace op
}  // namespace jax
}  // namespace frontend
}  // namespace ov
