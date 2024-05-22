// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/abs.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/shape_of.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

using namespace ov::op;

namespace {
OutputVector base_expand(const NodeContext& context, const Output<Node>& x, const Output<Node>& sizes) {
    auto shape = context.mark_node(std::make_shared<v0::Abs>(sizes));
    return {context.mark_node(std::make_shared<v3::Broadcast>(x, shape, BroadcastType::BIDIRECTIONAL))};
};
}  // namespace

OutputVector translate_expand(const NodeContext& context) {
    // aten::expand(Tensor(a) self, SymInt[] size, *, bool implicit=False) -> Tensor(a)
    num_inputs_check(context, 2, 3);
    auto x = context.get_input(0);
    auto sizes = context.get_input(1);
    // TODO: figure out what implicit means
    PYTORCH_OP_CONVERSION_CHECK(context.input_is_none(2) || context.const_input<bool>(2) == false,
                                "Unexpected value of implicit for expand operation");
    return base_expand(context, x, sizes);
};

OutputVector translate_expand_as(const NodeContext& context) {
    num_inputs_check(context, 2, 2);
    auto x = context.get_input(0);
    auto y = context.get_input(1);
    auto shape = context.mark_node(std::make_shared<v3::ShapeOf>(y, element::i32));
    return {context.mark_node(std::make_shared<v3::Broadcast>(x, shape, BroadcastType::BIDIRECTIONAL))};
};

OutputVector translate_expand_fx(const NodeContext& context) {
    auto num_inputs = context.get_input_size();
    num_inputs_check(context, 2, num_inputs);
    auto x = context.get_input(0);
    std::vector<int32_t> shape_vec;
    auto sizes = context.get_input(1);
    if (num_inputs != 2) {
        for (size_t i = 1; i < num_inputs; i++) {
            auto a = context.get_input_from_visible_context(i).get_node_shared_ptr();
            auto shape_input = context.get_input(static_cast<int>(i));
            if (std::dynamic_pointer_cast<ov::op::v0::Parameter>(a) ||
                shape_input.get_partial_shape().rank().is_dynamic() ||
                shape_input.get_partial_shape().rank().get_length() == 0) {
                shape_vec.push_back(-1);
            } else {
                auto val = context.const_input<int32_t>(i);
                shape_vec.push_back(val);
            }
        }
        sizes = ov::op::v0::Constant::create(element::i32, Shape{num_inputs - 1}, shape_vec);
    }
    return base_expand(context, x, sizes);
};

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
