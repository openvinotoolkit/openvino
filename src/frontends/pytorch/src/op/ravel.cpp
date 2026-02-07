// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/reshape.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

using namespace ov::op;

OutputVector translate_ravel(const NodeContext& context) {
    // aten::ravel(Tensor self) -> Tensor
    num_inputs_check(context, 1, 1);
    auto input = context.get_input(0);
    auto neg_1 = context.mark_node(v0::Constant::create(element::i64, Shape{1}, {-1}));
    return {context.mark_node(std::make_shared<v1::Reshape>(input, neg_1, false))};
};

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov