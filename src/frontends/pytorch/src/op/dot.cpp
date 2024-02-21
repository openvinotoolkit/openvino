// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/matmul.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

using namespace ov::op;

OutputVector translate_dot(const NodeContext& context) {
    num_inputs_check(context, 2, 3);
    // "aten::dot(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)"

    auto tensor1 = context.get_input(0);
    auto tensor2 = context.get_input(1);
    align_eltwise_input_types(context, tensor1, tensor2, true);

    auto dot_product = context.mark_node(std::make_shared<v0::MatMul>(tensor1, tensor2));

    if (!context.input_is_none(2)) {
        context.mutate_input(2, dot_product);
    }
    return {dot_product};
};

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
