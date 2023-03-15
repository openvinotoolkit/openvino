// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/mvn.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

using namespace ov::op;

OutputVector translate_layer_norm(NodeContext& context) {
    num_inputs_check(context, 5, 6);
    auto eps = context.const_input<float>(4);
    auto normalized_shape = context.const_input<Shape>(1);
    FRONT_END_OP_CONVERSION_CHECK(normalized_shape.size() == 1,
                                  "Translation for aten::layer_norm supports only single normalized_shape value, "
                                  "which means normalizing over the last dimension.");
    // TODO: support any dimention
    auto axes = context.mark_node(v0::Constant::create(element::i32, Shape{1}, {-1}));
    auto out_node =
        context.mark_node(std::make_shared<v6::MVN>(context.get_input(0), axes, true, eps, MVNEpsMode::INSIDE_SQRT));
    if (!context.input_is_none(2)) {
        out_node = context.mark_node(std::make_shared<v1::Multiply>(out_node, context.get_input(2)));
    }
    if (!context.input_is_none(3)) {
        out_node = context.mark_node(std::make_shared<v1::Add>(out_node, context.get_input(3)));
    }
    // Input with index 5 is flag "cudnn_enabled" we can ignore it
    return {out_node};
};

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov