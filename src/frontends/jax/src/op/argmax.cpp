// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/jax/node_context.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/op/topk.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace jax {
namespace op {

using namespace ov::op;

OutputVector translate_argmax(const NodeContext& context) {
    num_inputs_check(context, 1, 1);
    Output<Node> input = context.get_input(0);
    auto axis = context.const_named_param<int64_t>("axes");

    const auto k = v0::Constant::create(element::i64, Shape{}, {1});
    auto topk = std::make_shared<v11::TopK>(input, k, axis, v11::TopK::Mode::MAX, v1::TopK::SortType::NONE);
    auto indices = topk->output(1);

    auto squeeze_axis = v0::Constant::create(element::u64, Shape{}, {topk->get_axis()});
    auto res = std::make_shared<v0::Squeeze>(indices, squeeze_axis);

    return {res};
};

}  // namespace op
}  // namespace jax
}  // namespace frontend
}  // namespace ov