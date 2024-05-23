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
    auto reshape = std::make_shared<ov::op::v1::Reshape>(context.get_input(0), context.get_input(1), false);
    return {context.mark_node(reshape)};
};

OutputVector translate_reshape_fx(const NodeContext& context) {
    // Schema: aten.view.default(Tensor input, int[] shape) -> Tensor
    auto num_inputs = context.get_input_size();
    num_inputs_check(context, 2, num_inputs);
    std::vector<int32_t> shape_vec;
    if (context.get_input_type(1).is<type::List>()) {
        int num_dyn_dims = 0;
        for (size_t i = 1; i < num_inputs; i++) {
            auto shape_input = context.get_input(static_cast<int>(i));
            if (context.get_input_type(i).as<type::List>().element_type.is<type::PyScalar>()) {
                auto const_val = context.const_input<int32_t>(i);
                shape_vec.push_back(const_val);
            } else {
                // Set dimension to be dynamic if it's coming from an argument or another node
                shape_vec.push_back(-1);
                num_dyn_dims++;
            }
        }
        // We cannot use multiple -1s if there are more than 1 dynamic dimensions
        if (num_dyn_dims >= 2) {
            auto inp_shape = context.get_input(0).get_partial_shape();
            // If there are multiple dynamic dymensions, we cannot support inputs with dynamic rank
            if (inp_shape.rank().is_static()) {
                auto zero = context.mark_node(ov::op::v0::Constant::create(element::i32, Shape{1}, {0}));
                if (inp_shape.size() >= 3 && inp_shape.size() + 1 == shape_vec.size() && shape_vec[0] == 1 &&
                    inp_shape[0] == shape_vec[1]) {
                    // [N, ...] -> [1, N, ...] Can be translated to Unsqueeze
                    auto unsqueeze =
                        context.mark_node(std::make_shared<ov::op::v0::Unsqueeze>(context.get_input(0), zero));
                    return {unsqueeze};
                } else if (shape_vec.size() >= 3 && shape_vec.size() + 1 == inp_shape.size() && inp_shape[0] == 1 &&
                           inp_shape[1] == shape_vec[0]) {
                    // [1, N, ...] -> [N, ...] Can be translated to Squeeze
                    auto squeeze = context.mark_node(std::make_shared<ov::op::v0::Squeeze>(context.get_input(0), zero));
                    return {squeeze};
                } else if (inp_shape.size() == shape_vec.size()) {
                    // If the input rank is equal to output rank, we can use 0s in place of dynamic dimensions
                    for (size_t k = 0; k < shape_vec.size(); k++) {
                        if (shape_vec[k] == -1)
                            shape_vec[k] = 0;
                    }
                } else {
                    FRONT_END_GENERAL_CHECK(
                        false,
                        "Cannot support reshape with multiple dynamic dimensions for unequal ranks");
                }
            } else {
                FRONT_END_GENERAL_CHECK(
                    false,
                    "Cannot support reshape with multiple dynamic dimensions for dynamic input ranks");
            }
        }

        auto shape_const = ov::op::v0::Constant::create(element::i32, Shape{num_inputs - 1}, shape_vec);
        auto reshape = std::make_shared<ov::op::v1::Reshape>(context.get_input(0), shape_const, true);
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
