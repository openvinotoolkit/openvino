// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/erfinv.hpp"

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/convert.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

OutputVector translate_erfinv(const NodeContext& context) {
    // aten::erfinv(Tensor self) -> Tensor
    // aten::erfinv.out(Tensor self, Tensor(!a) out) -> Tensor(!a)
    num_inputs_check(context, 1, 2);
    auto x = context.get_input(0);
    auto xdtype = x.get_element_type();
    // in torch, erfinv returns always float dtype, while ov casts to input dtype
    if (xdtype.is_dynamic() || !xdtype.is_real()) {
        x = context.mark_node(std::make_shared<ov::op::v0::Convert>(x, element::f32));
    }

    auto y = context.mark_node(std::make_shared<ov::op::v17::ErfInv>(x));
    if (!context.input_is_none(1)) {
        context.mutate_input(1, y);
    }
    return {y};
};

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
