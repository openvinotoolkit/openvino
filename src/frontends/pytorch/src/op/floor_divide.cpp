// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/floor.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

using namespace ov::op;

OutputVector translate_floor_divide(const NodeContext& context) {
    std::cout << "Translating aten::floor_divide\n";
    num_inputs_check(context, 2, 2);
    Output<Node> x;
    Output<Node> y;
    std::tie(x, y) = get_inputs_with_promoted_types(context, 0, 1);

    // Store original type to convert back after floor division
    auto original_type = x.get_element_type();

    // If both inputs are integers, convert to f32 to avoid integer division truncation
    const auto x_dtype = x.get_element_type();
    const auto y_dtype = y.get_element_type();
    if (x_dtype.is_static() && x_dtype.is_integral() && y_dtype.is_static() && y_dtype.is_integral()) {
        std::cout << "In new code";
        x = context.mark_node(std::make_shared<v0::Convert>(x, element::f32));
        y = context.mark_node(std::make_shared<v0::Convert>(y, element::f32));
    }

    auto div = context.mark_node(std::make_shared<v1::Divide>(x, y, true));
    auto floor_result = context.mark_node(std::make_shared<v0::Floor>(div));

    // Convert back to original integer type if needed
    if (original_type.is_integral()) {
        floor_result = context.mark_node(std::make_shared<v0::Convert>(floor_result, original_type));
    }

    return {floor_result};
};

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
