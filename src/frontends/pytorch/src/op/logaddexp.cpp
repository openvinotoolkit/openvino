// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/convert_like.hpp"
#include "openvino/op/exp.hpp"
#include "openvino/op/log.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

using namespace ov::op;

OutputVector translate_logaddexp(const NodeContext& context) {
    // "aten::logaddexp(Tensor self, Tensor other) -> Tensor"
    num_inputs_check(context, 2, 2);

    auto input1 = context.get_input(0);
    auto input2 = context.get_input(1);

    // Convert inputs to floating point type if needed
    input1 = context.mark_node(std::make_shared<v0::Convert>(input1, element::f32));
    input2 = context.mark_node(std::make_shared<v0::Convert>(input2, element::f32));

    // Calculate exp(input1) and exp(input2)
    auto exp1 = context.mark_node(std::make_shared<v0::Exp>(input1));
    auto exp2 = context.mark_node(std::make_shared<v0::Exp>(input2));

    // Add the exponentials
    auto sum = context.mark_node(std::make_shared<v1::Add>(exp1, exp2));

    // Take the natural logarithm of the sum
    auto result = context.mark_node(std::make_shared<v0::Log>(sum));

    // If the output tensor type is different, convert to match
    if (input1.get_element_type() != element::f32) {
        result = context.mark_node(std::make_shared<v1::ConvertLike>(result, input1));
    }

    return {result};
};

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov