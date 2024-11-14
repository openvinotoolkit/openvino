// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/jax/node_context.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/reduce_max.hpp"
#include "openvino/op/reduce_sum.hpp"
#include "utils.hpp"

using namespace std;
using namespace ov;
using namespace ov::op;

namespace ov {
namespace frontend {
namespace jax {
namespace op {

template <typename T>
OutputVector translate_reduce_op(const NodeContext& context) {
    num_inputs_check(context, 1, 1);
    auto axes = context.const_named_param<vector<int64_t>>("axes");
    auto a = context.get_input(0);
    auto axes_const = make_shared<v0::Constant>(element::i64, Shape{axes.size()}, axes);
    Output<Node> reduce_op = make_shared<T>(a, axes_const, false);
    return {reduce_op};
}

template OutputVector translate_reduce_op<v1::ReduceMax>(const NodeContext& node);
template OutputVector translate_reduce_op<v1::ReduceSum>(const NodeContext& node);

}  // namespace op
}  // namespace jax
}  // namespace frontend
}  // namespace ov
