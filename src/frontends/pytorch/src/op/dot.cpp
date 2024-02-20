// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/opsets/opset8.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/reduce_sum.hpp"
#include "pt_framework_node.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

using namespace ov::op;

OutputVector translate_dot(const NodeContext& context) {
    num_inputs_check(context, 2, 3);
    // "aten::dot(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)"

    auto tensor1 = context.mark_node(std::make_shared<v0::Convert>(context.get_input(0), element::f32));
    auto tensor2 = context.mark_node(std::make_shared<v0::Convert>(context.get_input(1), element::f32));

    auto multiply = context.mark_node(std::make_shared<v1::Multiply>(tensor1, tensor2));
    auto axes = v0::Constant::create(element::i64, Shape{1}, {0});
    auto dot_product = context.mark_node(std::make_shared<v1::ReduceSum>(multiply, axes));

    if (!context.input_is_none(2)) {
        context.mutate_input(2, dot_product);
    }

    return {dot_product};
};

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov