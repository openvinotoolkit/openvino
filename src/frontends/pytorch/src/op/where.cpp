// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/non_zero.hpp"
#include "openvino/op/select.hpp"
#include "openvino/op/split.hpp"
#include "openvino/op/squeeze.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

using namespace ov::op;

OutputVector translate_where(const NodeContext& context) {
    num_inputs_check(context, 1, 3);
    auto cond = context.get_input(0);
    if (context.input_is_none(1)) {
        // aten::where(cond) is equivalent to torch.nonzero(cond, as_tuple=True):
        // returns a tuple of 1D index tensors, one per dimension of cond.
        const auto& cond_shape = cond.get_partial_shape();
        PYTORCH_OP_CONVERSION_CHECK(cond_shape.rank().is_static(),
                                    "aten::where(cond) with input of dynamic rank is not supported");
        const auto ndim = cond_shape.rank().get_length();
        auto non_zero = context.mark_node(std::make_shared<v3::NonZero>(cond));
        auto axis = context.mark_node(v0::Constant::create(element::i32, Shape{}, {0}));
        OutputVector result;
        if (ndim > 0) {
            auto split = context.mark_node(std::make_shared<v1::Split>(non_zero, axis, ndim));
            for (size_t i = 0; i < ndim; ++i) {
                result.push_back(context.mark_node(std::make_shared<v0::Squeeze>(split->output(i), axis)));
            }
        }
        return {context.mark_node(make_list_construct(result))};
    }
    auto bool_cond = context.mark_node(std::make_shared<v0::Convert>(cond, element::boolean));
    Output<Node> x;
    Output<Node> y;
    std::tie(x, y) = get_inputs_with_promoted_types(context, 1, 2);
    return {context.mark_node(std::make_shared<v1::Select>(bool_cond, x, y))};
};

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
