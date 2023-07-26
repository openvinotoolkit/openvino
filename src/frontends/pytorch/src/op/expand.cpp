// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/equal.hpp"
#include "openvino/op/select.hpp"
#include "openvino/op/shape_of.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

using namespace ov::op;

namespace {
OutputVector base_expand(const NodeContext& context, const Output<Node>& x, const Output<Node>& sizes) {
    auto one = context.mark_node(v0::Constant::create(element::i32, Shape{}, {1}));
    auto sizes_shape = context.mark_node(std::make_shared<v3::ShapeOf>(sizes, element::i32));
    auto neg_one = context.mark_node(v0::Constant::create(element::i32, Shape{}, {-1}));
    auto neg_ones = context.mark_node(std::make_shared<v3::Broadcast>(neg_one, sizes_shape));
    auto ones = context.mark_node(std::make_shared<v3::Broadcast>(one, sizes_shape));
    auto neg_sizes = context.mark_node(std::make_shared<v1::Equal>(sizes, neg_ones));
    auto shape = context.mark_node(std::make_shared<v1::Select>(neg_sizes, ones, sizes));
    return {context.mark_node(std::make_shared<v3::Broadcast>(x, shape, BroadcastType::BIDIRECTIONAL))};
};
}  // namespace

OutputVector translate_expand(const NodeContext& context) {
    // aten::expand(Tensor(a) self, SymInt[] size, *, bool implicit=False) -> Tensor(a)
    num_inputs_check(context, 2, 3);
    auto x = context.get_input(0);
    auto sizes = context.get_input(1);
    // TODO: figure out what implicit means
    FRONT_END_OP_CONVERSION_CHECK(context.input_is_none(2) || context.const_input<bool>(2) == false,
                                  "Unexpected value of implicit for expand operation");
    return base_expand(context, x, sizes);
};

OutputVector translate_expand_as(const NodeContext& context) {
    num_inputs_check(context, 2, 2);
    auto x = context.get_input(0);
    auto y = context.get_input(1);
    auto sizes = context.mark_node(std::make_shared<v3::ShapeOf>(y, element::i32));
    return base_expand(context, x, sizes);
};

OutputVector translate_expand_fx(const NodeContext& context) {
    // aten::expand(Tensor(a) self, SymInt[] size, *, bool implicit=False) -> Tensor(a)
    num_inputs_check(context, 2, 3);
    auto x = context.get_input(0);
    // TODO: This is a temporary solution to optimize out Broadcast if the input and
    // output shapes are same. This should be removed after a proper optimization is
    // implemented.
    auto sizes_const = context.const_input<Shape>(1);
    if (x.get_shape() == sizes_const) {
        return {x};
    }
    auto sizes = context.get_input(1);
    // TODO: figure out what implicit means
    FRONT_END_OP_CONVERSION_CHECK(context.input_is_none(2) || context.const_input<bool>(2) == false,
                                  "Unexpected value of implicit for expand operation");
    return base_expand(context, x, sizes);
};

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov