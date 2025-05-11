// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "default_opset.hpp"
#include "op_utils.hpp"
#include "openvino/frontend/paddle/node_context.hpp"

namespace ov {
namespace frontend {
namespace paddle {
namespace op {
NamedOutputs fill_any_like(const NodeContext& node) {
    auto x = node.get_input("X");
    auto dtype = node.get_attribute<ov::element::Type>("dtype", element::dynamic);
    const auto value = node.get_attribute<float>("value");
    if (dtype.is_dynamic()) {
        // when type does not define, use the input type
        dtype = x.get_element_type();
    }
    const std::vector<element::Type> supported_type =
        {element::boolean, element::i16, element::i32, element::i64, element::f16, element::f32, element::f64};
    const bool valid_type =
        std::any_of(supported_type.begin(), supported_type.end(), [dtype](const element::Type& type) {
            return dtype == type;
        });
    PADDLE_OP_CHECK(node, valid_type, "Invalid dtype! Fill_any_like supports boolean, i16, i32, i64, f16, f32, f64");
    const auto value_node = default_opset::Constant::create(dtype, {1}, {value});
    x = get_tensor_safe(x);
    const auto shape_node = std::make_shared<default_opset::ShapeOf>(x);
    return node.default_single_output_mapping({std::make_shared<default_opset::Broadcast>(value_node, shape_node)},
                                              {"Out"});
}

}  // namespace op
}  // namespace paddle
}  // namespace frontend
}  // namespace ov
