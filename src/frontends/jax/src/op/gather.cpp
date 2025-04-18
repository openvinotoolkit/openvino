// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/jax/node_context.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/gather.hpp"
#include "utils.hpp"

using namespace std;
using namespace ov;
using namespace ov::op;

namespace ov {
namespace frontend {
namespace jax {
namespace op {

OutputVector translate_gather(const NodeContext& context) {
    num_inputs_check(context, 2);

    Output<Node> inputs = context.get_input(0);
    Output<Node> indices = context.get_input(1);

    int64_t axis = context.const_named_param<int64_t>("axes");
    auto axis_node = make_shared<v0::Constant>(element::i64, Shape{}, axis);

    Output<Node> res = make_shared<v1::Gather>(inputs, indices, axis_node);

    return {res};
};

}  // namespace op
}  // namespace jax
}  // namespace frontend
}  // namespace ov