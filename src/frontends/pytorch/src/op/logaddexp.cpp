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
#include "openvino/op/maximum.hpp"
#include "openvino/op/subtract.hpp"
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

    // Find maximum of inputs
    auto max_val = context.mark_node(std::make_shared<v1::Maximum>(input1, input2));

    // Calculate x1 - max and x2 - max
    auto diff1 = context.mark_node(std::make_shared<v1::Subtract>(input1, max_val));
    auto diff2 = context.mark_node(std::make_shared<v1::Subtract>(input2, max_val));

    // Calculate exp(x1 - max) and exp(x2 - max)
    auto exp1 = context.mark_node(std::make_shared<v0::Exp>(diff1));
    auto exp2 = context.mark_node(std::make_shared<v0::Exp>(diff2));

    // Add the scaled exponentials
    auto sum = context.mark_node(std::make_shared<v1::Add>(exp1, exp2));

    // Take the log and add back the maximum
    auto log_sum = context.mark_node(std::make_shared<v0::Log>(sum));
    auto result = context.mark_node(std::make_shared<v1::Add>(log_sum, max_val));

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