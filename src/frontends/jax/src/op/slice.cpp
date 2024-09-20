// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/slice.hpp"

#include <memory>
#include <vector>

#include "openvino/core/node.hpp"
#include "openvino/core/node_output.hpp"
#include "openvino/frontend/jax/node_context.hpp"
#include "openvino/op/constant.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace jax {
namespace op {

using namespace ov::op;

OutputVector translate_slice(const NodeContext& context) {
    num_inputs_check(context, 1, 1);
    Output<Node> input = context.get_input(0);
    auto start_indices = context.const_named_param<std::shared_ptr<v0::Constant>>("start_indices");
    auto limit_indices = context.const_named_param<std::shared_ptr<v0::Constant>>("limit_indices");

    Output<Node> strides;
    if (context.has_param("strides")) {
        strides = context.const_named_param<std::shared_ptr<v0::Constant>>("strides");
    } else {
        strides = std::make_shared<op::v0::Constant>(element::i64, start_indices->get_shape(), 1);
    }
    Output<Node> res = std::make_shared<v8::Slice>(input, start_indices, limit_indices, strides);
    return {res};
};

}  // namespace op
}  // namespace jax
}  // namespace frontend
}  // namespace ov