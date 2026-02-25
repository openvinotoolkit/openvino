// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/add.hpp"
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
    // "aten::logaddexp(Tensor self, Tensor other, out=None) -> Tensor"
    num_inputs_check(context, 2, 3);
    Output<Node> input1;
    Output<Node> input2;
    std::tie(input1, input2) = get_inputs_with_promoted_types(context, 0, 1);
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

    if (!context.input_is_none(2)) {
        context.mutate_input(2, result);
    }

    return {result};
};

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov