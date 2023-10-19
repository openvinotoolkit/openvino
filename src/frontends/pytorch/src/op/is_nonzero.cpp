// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/equal.hpp"
#include "openvino/op/logical_and.hpp"
#include "openvino/op/logical_or.hpp"
#include "openvino/op/not_equal.hpp"
#include "openvino/op/shape_of.hpp"
#include "pt_framework_node.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

using namespace ov::op;

OutputVector translate_is_nonzero(const NodeContext& context) {
    num_inputs_check(context, 1, 1);
    auto input = context.get_input(0);

    auto zero_tensor = context.mark_node(v0::Constant::create(element::f32, Shape{1}, {0.0}));
    auto one_tensor = context.mark_node(v0::Constant::create(element::i64, Shape{1}, {1}));
    auto false_tensor = context.mark_node(v0::Constant::create(element::boolean, Shape{1}, {false}));

    // check if length input is 1
    auto input_shape = context.mark_node(std::make_shared<v3::ShapeOf>(input));
    auto is_length_one = context.mark_node(std::make_shared<v1::Equal>(input_shape, one_tensor));

    // perform type conversion
    auto converted_input = context.mark_node(std::make_shared<v0::Convert>(input, element::f32));

    auto is_nonzero_numeric = context.mark_node(std::make_shared<v1::NotEqual>(converted_input, zero_tensor));
    auto is_nonzero_boolean = context.mark_node(std::make_shared<v1::NotEqual>(input, false_tensor));

    auto final_result = context.mark_node(std::make_shared<v1::LogicalAnd>(
        is_length_one,
        context.mark_node(std::make_shared<v1::LogicalOr>(is_nonzero_numeric, is_nonzero_boolean))));

    return {final_result};
};

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
