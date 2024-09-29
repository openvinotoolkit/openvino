// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/jax/node_context.hpp"
#include "openvino/op/equal.hpp"
#include "openvino/op/greater.hpp"
#include "openvino/op/greater_eq.hpp"
#include "openvino/op/less.hpp"
#include "openvino/op/less_eq.hpp"
#include "openvino/op/not_equal.hpp"
#include "utils.hpp"

using namespace std;
using namespace ov;
using namespace ov::op;

namespace ov {
namespace frontend {
namespace jax {
namespace op {

template <typename T>
OutputVector translate_binary_op(const NodeContext& context) {
    num_inputs_check(context, 2, 2);
    auto lhs = context.get_input(0);
    auto rhs = context.get_input(1);
    Output<Node> binary_op = make_shared<T>(lhs, rhs);
    return {binary_op};
}

template OutputVector translate_binary_op<v1::Equal>(const NodeContext& context);
template OutputVector translate_binary_op<v1::GreaterEqual>(const NodeContext& context);
template OutputVector translate_binary_op<v1::Greater>(const NodeContext& context);
template OutputVector translate_binary_op<v1::Less>(const NodeContext& context);
template OutputVector translate_binary_op<v1::LessEqual>(const NodeContext& context);
template OutputVector translate_binary_op<v1::NotEqual>(const NodeContext& context);

}  // namespace op
}  // namespace jax
}  // namespace frontend
}  // namespace ov
