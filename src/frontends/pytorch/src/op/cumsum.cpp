// Copyright (C) 2018-2023 Intel Corporation
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

OutputVector translate_cumsum(NodeContext& context) {
    // aten::cumsum(Tensor self, int dim, *, ScalarType? dtype=None, Tensor out=None)
    num_inputs_check(context, 2, 4);
    auto x = context.get_input(0);
    auto dim = context.get_input(1);
    if (!context.input_is_none(2)) {
        auto dtype = convert_dtype(context.const_input<int64_t>(2));
        x = context.mark_node(std::make_shared<ov::op::v0::Convert>(x, dtype));
    }
    return {context.mark_node(std::make_shared<ov::op::v0::CumSum>(x, dim))};
};

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
