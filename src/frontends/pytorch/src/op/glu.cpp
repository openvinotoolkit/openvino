// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/sigmoid.hpp"
#include "openvino/op/split.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

using namespace ov::op;

OutputVector translate_glu(const NodeContext& context) {
    num_inputs_check(context, 1, 2);
    auto x = context.get_input(0);
    auto dim = context.input_is_none(1) ? context.mark_node(v0::Constant::create(element::i32, Shape{}, {-1}))
                                        : context.get_input(1);
    auto split = context.mark_node(std::make_shared<v1::Split>(x, dim, 2));
    auto first = split->output(0);
    auto second = split->output(1);
    auto sigmoid = context.mark_node(std::make_shared<v0::Sigmoid>(second));
    return {context.mark_node(std::make_shared<v1::Multiply>(first, sigmoid))};
};

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov