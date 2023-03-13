// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/convert_like.hpp"
#include "openvino/op/cum_sum.hpp"
#include "pt_framework_node.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

using namespace ov::op;

OutputVector translate_cumsum(NodeContext& context) {
    // aten::cumsum(Tensor self, int dim, *, ScalarType? dtype=None, Tensor out=None)
    num_inputs_check(context, 2, 4);
    auto x = context.get_input(0);
    auto dim = context.get_input(1);
    if (!context.input_is_none(2)) {
        if (std::dynamic_pointer_cast<v0::Constant>(context.get_input_from_visible_context(2).get_node_shared_ptr())) {
            auto dtype = convert_dtype(context.const_input<int64_t>(2));
            x = context.mark_node(std::make_shared<v0::Convert>(x, dtype));
        } else if (const auto& fw_node = cast_fw_node(context.get_input(2).get_node_shared_ptr(), "prim::dtype")) {
            auto out_tensor = fw_node->input_value(0);
            x = context.mark_node(std::make_shared<v1::ConvertLike>(x, out_tensor));
        } else {
            FRONT_END_OP_CONVERSION_CHECK(false, "Couldn't get dtype input");
        }
    }
    auto result = context.mark_node(std::make_shared<v0::CumSum>(x, dim));
    if (!context.input_is_none(3)) {
        context.mutate_input(3, result);
    }
    return {result};
};

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
