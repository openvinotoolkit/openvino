// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/opsets/opset8.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

OutputVector translate_arange(NodeContext& context) {
    auto zero = context.mark_node(opset8::Constant::create(element::i32, Shape{}, {0}));
    auto one = context.mark_node(opset8::Constant::create(element::i32, Shape{}, {1}));
    auto dtype = element::f32;
    bool dtype_applied = false;
    int num_inputs = context.get_input_size();
    // aten::arange(Scalar end, tensor out)
    if (num_inputs == 2) {
        auto end = context.get_input(0);
        auto range = context.mark_node(std::make_shared<opset8::Range>(zero, end, one, dtype));
        return {context.mark_node(std::make_shared<opset8::ConvertLike>(range, end))};
    }
    // # aten::arange(Scalar start, Scalar end, Scalar step, Tensor out)
    if (num_inputs == 4) {
        auto start = context.get_input(0);
        auto end = context.get_input(1);
        auto step = context.get_input(2);
        auto range = context.mark_node(std::make_shared<opset8::Range>(start, end, step, dtype));
        return {context.mark_node(std::make_shared<opset8::ConvertLike>(range, end))};
    }
    // aten::arange(Scalar end, ScalarType dtype, Layout, Device, bool pin_memory)
    if (num_inputs == 5) {
        auto end = context.get_input(0);
        if (!context.input_is_none(1)) {
            auto pt_type = context.const_input<int64_t>(1);
            FRONT_END_OP_CONVERSION_CHECK(TORCH_TO_OV_TYPE.count(pt_type), "Unknown type in aten::arange: ", pt_type);
            dtype = TORCH_TO_OV_TYPE.at(pt_type);
            end = context.mark_node(std::make_shared<opset8::Convert>(end, dtype));
            zero = context.mark_node(std::make_shared<opset8::Convert>(zero, dtype));
            one = context.mark_node(std::make_shared<opset8::Convert>(one, dtype));
            dtype_applied = true;
        }
        auto range = context.mark_node(std::make_shared<opset8::Range>(zero, end, one, dtype));
        if (!dtype_applied) {
            return {context.mark_node(std::make_shared<opset8::ConvertLike>(range, end))};
        }
        return {range};
    }
    // aten::arange(Scalar start, Scalar end, ScalarType dtype, Layout, Device, bool pin_memory)
    if (num_inputs == 6) {
        auto start = context.get_input(0);
        auto end = context.get_input(1);
        if (!context.input_is_none(2)) {
            auto pt_type = context.const_input<int64_t>(2);
            FRONT_END_OP_CONVERSION_CHECK(TORCH_TO_OV_TYPE.count(pt_type), "Unknown type in aten::arange: ", pt_type);
            dtype = TORCH_TO_OV_TYPE.at(pt_type);
            dtype_applied = true;
            end = context.mark_node(std::make_shared<opset8::Convert>(end, dtype));
            start = context.mark_node(std::make_shared<opset8::Convert>(start, dtype));
            one = context.mark_node(std::make_shared<opset8::Convert>(one, dtype));
        }
        auto range = context.mark_node(std::make_shared<opset8::Range>(start, end, one, dtype));
        if (!dtype_applied) {
            return {context.mark_node(std::make_shared<opset8::ConvertLike>(range, end))};
        }
        return {range};
    }
    // aten::arange(Scalar start, Scalar end, Scalar step, ScalarType dtype, Layout, Device, bool pin_memory)
    if (num_inputs == 7) {
        auto start = context.get_input(0);
        auto end = context.get_input(1);
        auto step = context.get_input(2);
        if (!context.input_is_none(3)) {
            auto pt_type = context.const_input<int64_t>(3);
            FRONT_END_OP_CONVERSION_CHECK(TORCH_TO_OV_TYPE.count(pt_type), "Unknown type in aten::arange: ", pt_type);
            dtype = TORCH_TO_OV_TYPE.at(pt_type);
            end = context.mark_node(std::make_shared<opset8::Convert>(end, dtype));
            start = context.mark_node(std::make_shared<opset8::Convert>(start, dtype));
            step = context.mark_node(std::make_shared<opset8::Convert>(step, dtype));
            dtype_applied = true;
        }
        auto range = context.mark_node(std::make_shared<opset8::Range>(start, end, step, dtype));
        if (!dtype_applied) {
            return {context.mark_node(std::make_shared<opset8::ConvertLike>(range, end))};
        }
        return {range};
    }
};

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov