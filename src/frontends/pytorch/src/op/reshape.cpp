// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/reshape.hpp"

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

OutputVector translate_reshape(const NodeContext& context) {
    // Translation is used by both aten::view and aten::reshape.
    // Schema: aten::view(Tensor input, int[] shape) -> Tensor
    // Schema: aten::reshape(Tensor input, int[] shape) -> Tensor
    // For shape parameter, int[] is converted into single dimensional Tensor.
    num_inputs_check(context, 2, 2);
    auto shape = get_input_concat_if_list(context, 1);
    auto reshape = std::make_shared<ov::op::v1::Reshape>(context.get_input(0), shape, false);
    return {context.mark_node(reshape)};
};

OutputVector translate_reshape_fx(const NodeContext& context) {
    // Schema: aten.view.default(Tensor input, int[] shape) -> Tensor
    auto num_inputs = context.get_input_size();
    num_inputs_check(context, 2, num_inputs);
    std::vector<int32_t> shape_vec;
    if (context.get_input_type(1).is<type::List>()) {
        auto concat = concat_list_from_inputs(context, 1, num_inputs);
        auto reshape = std::make_shared<ov::op::v1::Reshape>(context.get_input(0), concat, true);
        return {context.mark_node(reshape)};
    } else {
        auto shape_input = context.get_input(1);
        if (shape_input.get_partial_shape().rank().is_dynamic() ||
            shape_input.get_partial_shape().rank().get_length() == 0) {
            shape_vec.push_back(0);
            auto shape_const = ov::op::v0::Constant::create(element::i32, Shape{1}, shape_vec);
            auto result =
                context.mark_node(std::make_shared<ov::op::v1::Reshape>(context.get_input(0), shape_const, true));
            return {result};
        }
        auto reshape = std::make_shared<ov::op::v1::Reshape>(context.get_input(0), context.get_input(1), true);
        return {context.mark_node(reshape)};
    }
};

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
