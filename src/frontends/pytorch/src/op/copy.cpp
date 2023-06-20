// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/convert_like.hpp"
#include "openvino/op/shape_of.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

using namespace ov::op;

OutputVector translate_copy_(const NodeContext& context) {
    // aten::copy_(Tensor(a!) self, Tensor src, bool non_blocking=False) -> Tensor(a!)
    num_inputs_check(context, 2, 3);
    auto self = context.get_input(0);
    auto src = context.get_input(1);
    // Convert src to type of self
    auto src_converted = context.mark_node(std::make_shared<v1::ConvertLike>(src, self));
    // Broadcast src to shape of self
    auto self_shape = context.mark_node(std::make_shared<v3::ShapeOf>(self));
    Output<Node> res = context.mark_node(std::make_shared<v3::Broadcast>(src_converted, self_shape));
    context.mutate_input(0, res);
    return {res};
};

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov