// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/gather.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

OutputVector translate_embedding(const NodeContext& context) {
    // aten::embedding(Tensor weight, Tensor indices, SymInt padding_idx=-1, bool scale_grad_by_freq=False, bool
    // sparse=False)
    num_inputs_check(context, 2, 5);
    auto data = context.get_input(0);
    auto indices = context.get_input(1);
    indices = context.mark_node(std::make_shared<ov::op::v0::Convert>(indices, element::i32));
    // skip parameters 2, 3, 4 used only during training:
    // padding_idx - if specified, the entries at padding_idx do not contribute to the gradient
    // scale_grad_by_freq - if given, this will scale gradients by the inverse of frequency of
    //                      the words in the mini-batch.
    // sparse - if True, gradient will be represented as sparse tensor
    auto axis_0 = context.mark_node(ov::op::v0::Constant::create(element::i32, Shape{}, {0}));
    return {context.mark_node(std::make_shared<ov::op::v8::Gather>(data, indices, axis_0))};
};

OutputVector translate_embedding_ext(const NodeContext& context) {
    // used in 16-bit patching
    num_inputs_check(context, 2, 5);
    auto data = context.get_input(0);
    if (data.get_element_type() != element::f32) {
        data = context.mark_node(std::make_shared<ov::op::v0::Convert>(data, element::f32));
    }
    auto indices = context.get_input(1);
    indices = context.mark_node(std::make_shared<ov::op::v0::Convert>(indices, element::i32));
    auto axis_0 = context.mark_node(ov::op::v0::Constant::create(element::i32, Shape{}, {0}));
    return {context.mark_node(std::make_shared<ov::op::v8::Gather>(data, indices, axis_0))};
};

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
