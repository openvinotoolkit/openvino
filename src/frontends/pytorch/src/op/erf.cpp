// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/erf.hpp"

#include "openvino/frontend/pytorch/node_context.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

OutputVector translate_erf(const NodeContext& context) {
    // aten::erf(Tensor self) -> Tensor
    // aten::erf.out(Tensor self, Tensor(!a) out) -> Tensor(!a)
    num_inputs_check(context, 1, 2);
    auto x = context.get_input(0);
    
    auto y = context.mark_node(std::make_shared<ov::op::v0::Erf>(x));
    if (!context.input_is_none(1)){
        context.mutate_input(1, y);
    }
    return {y};
};

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov