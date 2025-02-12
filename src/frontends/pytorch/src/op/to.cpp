// Copyright (C) 2018-2025 Intel Corporation
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

OutputVector translate_to(const NodeContext& context) {
    int dtype_idx;
    if (context.get_input_size() == 5) {
        // aten::to.dtype(Tensor(a) self, int dtype, bool non_blocking=False, bool copy=False, int? memory_format=None)
        // -> (Tensor(a))
        dtype_idx = 1;
        auto node = context.get_input_from_visible_context(dtype_idx).get_node_shared_ptr();
        auto fw_node = ov::as_type_ptr<PtFrameworkNode>(node);
        if (fw_node && fw_node->get_op_type() == "prim::device") {
            // Cast only to device without changing dtype. Return input node unchanged.
            return {context.get_input(0)};
        }
        if (fw_node && fw_node->get_op_type() == "prim::Constant") {
            // Device param can be set using constant.
            if (!context.get_input_type(dtype_idx).is<type::Tensor>()) {
                // Cast only to device without changing dtype. Return input node unchanged.
                return {context.get_input(0)};
            }
        }

    } else if (context.get_input_size() == 6) {
        // aten::to.device(Tensor(a) self, Device device, int dtype, bool non_blocking=False, bool copy=False, int?
        // memory_format=None) -> (Tensor(a)).
        // Input with index 1 is device we skip that input.
        dtype_idx = 2;
        if (context.input_is_none(dtype_idx)) {
            // Cast only to device without changing dtype. Return input node unchanged.
            return {context.get_input(0)};
        }
    } else if (context.get_input_size() == 8) {
        // aten::to(Tensor(a) self, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool?
        // pin_memory=None, bool non_blocking=False, bool copy=False, MemoryFormat? memory_format=None)
        dtype_idx = 1;
        if (context.input_is_none(dtype_idx)) {
            // Cast only to device without changing dtype. Return input node unchanged.
            return {context.get_input(0)};
        }
    } else {
        PYTORCH_OP_CONVERSION_CHECK(false, "Unknown aten::to format");
    }

    // We ignore both non_blocking and copy inputs since non_blocking argument is used
    // in Pytorch during training to overlap data transfer from CPU to GPU which does
    // not have a use case in OV. To copy or not to copy inputs should not be set
    // on the frontend level since it can produce unexpected behaviour in the later
    // stages. (e.g. transformations passes)

    // memory_format sets the desired memory format of returned Tensor.
    // memory format is ignored since it changes strides of a tensor. In openvino tensors are always contigious
    auto dtype_ext_node = context.get_input_from_visible_context(dtype_idx).get_node_shared_ptr();
    auto dtype_fw_node = ov::as_type_ptr<PtFrameworkNode>(dtype_ext_node);
    Output<Node> cast;
    if (dtype_fw_node && dtype_fw_node->get_op_type() == "prim::dtype") {
        auto type_input = dtype_fw_node->input_value(0);
        cast = context.mark_node(std::make_shared<v1::ConvertLike>(context.get_input(0), type_input));
    } else if (const auto dtype_const = ov::as_type_ptr<v0::Constant>(dtype_ext_node)) {
        auto pt_type = dtype_const->cast_vector<int64_t>()[0];
        auto dtype = convert_dtype(pt_type);
        cast = context.mark_node(std::make_shared<v0::Convert>(context.get_input(0), dtype));
    } else {
        cast = context.mark_node(std::make_shared<v1::ConvertLike>(context.get_input(0), context.get_input(1)));
    }
    return {cast};
}

OutputVector translate_to_fx(const NodeContext& context) {
    num_inputs_check(context, 1, 1);
    auto data = context.get_input(0);
    if (context.has_attribute("dtype")) {
        auto dtype = context.get_attribute<element::Type>("dtype");
        data = context.mark_node(std::make_shared<v0::Convert>(context.get_input(0), dtype));
    }
    return {data};
}

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
