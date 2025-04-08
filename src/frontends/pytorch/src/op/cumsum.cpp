// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/cum_sum.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

using namespace ov::op;

OutputVector translate_cumsum(const NodeContext& context) {
    // aten::cumsum(Tensor self, int dim, *, ScalarType? dtype=None, Tensor out=None)
    num_inputs_check(context, 2, 4);
    auto x = context.get_input(0);
    auto dim = context.get_input(1);
    if (!context.input_is_none(2)) {
        x = apply_dtype(context, 2, x);
    }
    auto result = context.mark_node(std::make_shared<v0::CumSum>(x, dim));
    if (!context.input_is_none(3)) {
        context.mutate_input(3, result);
    }
    return {result};
};

OutputVector translate_cumsum_fx(const NodeContext& context) {
    // cumsum = torch.ops.aten.cumsum.default(arg0_1, 0, dtype = torch.float64)
    num_inputs_check(context, 2, 2);
    auto x = context.get_input(0);
    auto dim = context.get_input(1);
    if (context.has_attribute("dtype")) {
        auto dtype = context.get_attribute<element::Type>("dtype");
        x = context.mark_node(std::make_shared<v0::Convert>(x, dtype));
    }
    auto result = context.mark_node(std::make_shared<v0::CumSum>(x, dim));
    return {result};
};

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
