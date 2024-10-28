// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/convert_like.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/subtract.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

using namespace ov::op;

OutputVector translate_lerp(const NodeContext& context) {
    // Tensor = aten::lerp(%lhs.1, %rhs.1, %self.weight)
    num_inputs_check(context, 3, 3);
    Output<Node> start;
    Output<Node> end;
    std::tie(start, end) = get_inputs_with_promoted_types(context, 0, 1);

    Output<Node> weight = context.get_input(2);
    auto scale = context.mark_node(std::make_shared<v1::Subtract>(end, start));
    weight = context.mark_node(std::make_shared<v1::ConvertLike>(weight, scale));
    auto delta = context.mark_node(std::make_shared<v1::Multiply>(scale, weight));
    return {context.mark_node(std::make_shared<v1::Add>(start, delta))};
};

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
