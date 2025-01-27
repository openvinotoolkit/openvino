// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/exp.hpp"
#include "openvino/op/log.hpp"
#include "openvino/op/maximum.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/constant.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

OutputVector translate_contains(const NodeContext& context) {
    // "aten::__contains__(Tensor self, Scalar value) -> bool"
    num_inputs_check(context, 2, 2);

    auto input_tensor = context.get_input(0);
    auto scalar_value = context.get_input(1);

    ''' Check if the scalar value needs to be converted to the tensor type '''
    scalar_value = context.mark_node(std::make_shared<v0::Convert>(scalar_value, input_tensor.get_element_type()));

    ''' Cast Broadcast the scalar to the same shape as input_tensor (translation) '''
    auto broadcast_scalar = context.mark_node(std::make_shared<v0::Broadcast>(scalar_value, input_tensor.get_shape()));

    auto comparison_result = context.mark_node(std::make_shared<v0::Equal>(input_tensor, broadcast_scalar));

    ''' Reduce the comparison result by summing along the relevant axis to check for any match '''
    auto reduction = context.mark_node(std::make_shared<v0::ReduceSum>(comparison_result, {0}, true));

    auto result = context.mark_node(std::make_shared<v0::Greater>(reduction, v0::Constant::create(element::i32, Shape{}, {0})));

    return {result};
};

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
