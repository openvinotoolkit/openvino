// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/convert_like.hpp"
#include "pt_framework_node.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

using namespace ov::op;

OutputVector translate_as_tensor(NodeContext& context) {
    // aten::tensor(t[] data, *, ScalarType? dtype=None, Device? device=None, bool requires_grad=False) -> Tensor
    num_inputs_check(context, 1, 4);
    auto dtype = element::f32;
    if (!context.input_is_none(1)) {
        auto dtype_ext_node = context.get_input_from_visible_context(1).get_node_shared_ptr();
        auto dtype_fw_node = std::dynamic_pointer_cast<PtFrameworkNode>(dtype_ext_node);
        if (dtype_fw_node && dtype_fw_node->get_op_type() == "prim::dtype") {
            auto type_input = dtype_fw_node->input_value(0);
            return {context.mark_node(std::make_shared<v1::ConvertLike>(context.get_input(0), type_input))};
        }
        if (auto dtype_const = std::dynamic_pointer_cast<v0::Constant>(dtype_ext_node)) {
            auto pt_type = dtype_const->cast_vector<int64_t>()[0];
            dtype = convert_dtype(pt_type);
        }
    }
    auto cast = context.mark_node(std::make_shared<v0::Convert>(context.get_input(0), dtype));

    // Input with index 2 is device, we skip this input
    // Input with index 3 is flag requires_grad, we skip this input
    return {cast};
};

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov