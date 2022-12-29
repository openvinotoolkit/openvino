// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/opsets/opset8.hpp"
#include "pt_framework_node.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

OutputVector translate_as_tensor(NodeContext& context) {
    auto dtype = element::f32;
    Output<Node> cast;
    if (!context.input_is_none(1)) {
        auto dtype_ext_node = context.get_input_from_visible_context(1).get_node_shared_ptr();
        auto dtype_fw_node = std::dynamic_pointer_cast<PtFrameworkNode>(dtype_ext_node);
        if (dtype_fw_node && dtype_fw_node->get_op_type() == "prim::dtype") {
            auto type_input = dtype_fw_node->input_value(0);
            return {context.mark_node(std::make_shared<opset8::ConvertLike>(context.get_input(0), type_input))};
        }
        if (auto dtype_const = std::dynamic_pointer_cast<opset8::Constant>(dtype_ext_node)) {
            auto pt_type = dtype_const->cast_vector<int64_t>()[0];
            FRONT_END_OP_CONVERSION_CHECK(TORCH_TO_OV_TYPE.count(pt_type),
                                          "Unknown type in aten::as_tensor: ",
                                          pt_type);
            dtype = TORCH_TO_OV_TYPE.at(pt_type);
        }
    }
    cast = context.mark_node(std::make_shared<opset8::Convert>(context.get_input(0), dtype));

    // Input with index 2 is device, we skip this input
    return {cast};
};

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov