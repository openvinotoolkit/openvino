// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/opsets/opset8.hpp"
#include "op_table.hpp"

#include "pt_framework_node.hpp"


namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

const std::map<int, element::Type> type_map{
    {0, element::u8},
    {1, element::i8},
    {2, element::i16},
    {3, element::i32},
    {4, element::i64},
    {5, element::f16},
    {6, element::f32},
    {7, element::f64},
};


OutputVector translate_aten_to(NodeContext& context) {
    int dtype_idx;
    int non_blocking_idx;
    int copy_idx;
    int memory_format_idx;
    if (context.get_input_size() == 5) {
        // aten::to.dtype(Tensor(a) self, int dtype, bool non_blocking=False, bool copy=False, int? memory_format=None)
        // -> (Tensor(a))
        dtype_idx = 1;
        non_blocking_idx = 2;
        copy_idx = 3;
        memory_format_idx = 4;
    } else if (context.get_input_size() == 6) {
        // aten::to.device(Tensor(a) self, Device device, int dtype, bool non_blocking=False, bool copy=False, int?
        // memory_format=None) -> (Tensor(a)).
        // Skipping "device" input.
        dtype_idx = 2;
        non_blocking_idx = 3;
        copy_idx = 4;
        memory_format_idx = 5;
    } else {
        FRONT_END_OP_CONVERSION_CHECK(false, "Unknown aten::to format");
    }
    // TODO: do we need to check these inputs?
    // OV_FRONTEND_REQUIRE(context.const_input<bool>(non_blocking_idx) == false);
    // OV_FRONTEND_REQUIRE(context.const_input<bool>(copy_idx) == false);
    // OV_FRONTEND_REQUIRE(context.input_is_none(memory_format_idx));
    auto dtype_ext_node = context.get_input_from_visible_context(dtype_idx).get_node_shared_ptr();
    auto dtype_tensor = context.get_input(dtype_idx);
    auto dtype_fw_node = std::dynamic_pointer_cast<PtFrameworkNode>(dtype_tensor.get_node_shared_ptr());
    Output<Node> cast;
    if (dtype_fw_node && dtype_fw_node->get_op_type() == "prim::dtype") {
        auto type_input = dtype_fw_node->input(0).get_source_output();
        cast = context.mark_node(std::make_shared<opset8::ConvertLike>(context.get_input(0), type_input));
    } else if (std::dynamic_pointer_cast<opset8::Constant>(dtype_ext_node)) {
        auto pt_type = context.const_input<int64_t>(dtype_idx);
        FRONT_END_OP_CONVERSION_CHECK(op::type_map.count(pt_type), "Unknown type in aten::to: ", pt_type);
        auto dtype = type_map.at(pt_type);
        cast = context.mark_node(std::make_shared<opset8::Convert>(context.get_input(0), dtype));
    } else {
        cast = context.mark_node(std::make_shared<opset8::ConvertLike>(context.get_input(0), dtype_tensor));
    }
    return {cast};
}

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov