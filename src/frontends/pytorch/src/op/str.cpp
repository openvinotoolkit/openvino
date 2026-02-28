// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/constant.hpp"
#include "pt_framework_node.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

OutputVector translate_str(const NodeContext& context) {
    // aten::str(t elem) -> str
    num_inputs_check(context, 1, 1);
    auto input_node = context.get_input(0).get_node_shared_ptr();
    auto const_node = ov::as_type_ptr<ov::op::v0::Constant>(input_node);

    if (const_node) {
        std::string str_value;
        auto dtype = const_node->get_element_type();
        if (dtype == element::i64) {
            str_value = std::to_string(const_node->cast_vector<int64_t>()[0]);
        } else if (dtype == element::i32) {
            str_value = std::to_string(const_node->cast_vector<int32_t>()[0]);
        } else if (dtype == element::f32) {
            str_value = std::to_string(const_node->cast_vector<float>()[0]);
        } else if (dtype == element::f64) {
            str_value = std::to_string(const_node->cast_vector<double>()[0]);
        } else if (dtype == element::boolean) {
            str_value = const_node->cast_vector<bool>()[0] ? "True" : "False";
        } else {
            return {context.mark_node(std::make_shared<PtFrameworkNode>(context.get_decoder(), context.inputs()))};
        }
        // Return a 1D i64 constant with shape {str_len} so that aten::len can read
        // back the string length via ShapeOf without needing runtime string tensors.
        auto str_len = str_value.size();
        auto result_const =
            ov::op::v0::Constant::create(element::i64, Shape{str_len}, std::vector<int64_t>(str_len, 0));
        return {context.mark_node(result_const)};
    }

    return {context.mark_node(std::make_shared<PtFrameworkNode>(context.get_decoder(), context.inputs()))};
}

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
