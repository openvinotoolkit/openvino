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

OutputVector translate_full(NodeContext& context) {
    auto sizes = context.get_input(0);
    auto value = context.get_input(1);
    int num_inputs = context.get_input_size();

    auto filled_tensor = context.mark_node(std::make_shared<opset8::Broadcast>(value, sizes));
    if (num_inputs < 6) {
        size_t out_id = num_inputs == 3 ? 2: 3;
        if (!context.input_is_none(out_id)){
        auto out = context.get_input(out_id);
        return {context.mark_node(std::make_shared<opset8::ConvertLike>(filled_tensor, out))};
        }
    }
    size_t dtype_id = num_inputs == 6 ? 2: 3;
    if (!context.input_is_none(dtype_id)){
        auto pt_type = context.const_input<int64_t>(dtype_id);
        FRONT_END_OP_CONVERSION_CHECK(TORCH_TO_OV_TYPE.count(pt_type), "Unknown type in aten::full: ", pt_type);
        auto dtype = TORCH_TO_OV_TYPE.at(pt_type);
        filled_tensor = context.mark_node(std::make_shared<opset8::Convert>(filled_tensor, dtype));
    }
    return {filled_tensor};
};

OutputVector translate_full_like(NodeContext& context) {
    auto input = context.get_input(0);
    auto value = context.get_input(1);
    auto input_shape = context.mark_node(std::make_shared<opset8::ShapeOf>(input));
    auto filled_tensor = context.mark_node(std::make_shared<opset8::Broadcast>(value, input_shape));
    if (context.get_input_size() == 7 && !context.input_is_none(2)){
        auto pt_type = context.const_input<int64_t>(2);
        FRONT_END_OP_CONVERSION_CHECK(TORCH_TO_OV_TYPE.count(pt_type), "Unknown type in aten::full_like: ", pt_type);
        auto dtype = TORCH_TO_OV_TYPE.at(pt_type);
        filled_tensor = context.mark_node(std::make_shared<opset8::Convert>(filled_tensor, dtype));
    } else {
        auto out_dtype = context.input_is_none(3)? input : context.get_input(3);
        filled_tensor = context.mark_node(std::make_shared<opset8::ConvertLike>(filled_tensor, out_dtype));
    }
    return {filled_tensor};
};

OutputVector translate_new_full(NodeContext& context) {
    auto input = context.get_input(0);
    auto sizes = context.get_input(1);
    auto value = context.get_input(2);
    auto filled_tensor = context.mark_node(std::make_shared<opset8::Broadcast>(value, sizes));
    if (context.get_input_size() == 7 && !context.input_is_none(3)) {
        auto pt_type = context.const_input<int64_t>(3);
        FRONT_END_OP_CONVERSION_CHECK(TORCH_TO_OV_TYPE.count(pt_type), "Unknown type in aten::new_full: ", pt_type);
        auto dtype = TORCH_TO_OV_TYPE.at(pt_type);
        return {context.mark_node(std::make_shared<opset8::Convert>(filled_tensor, dtype))};
    }
    return {context.mark_node(std::make_shared<opset8::ConvertLike>(filled_tensor, input))};
};

OutputVector translate_zeros(NodeContext& context) {
    auto sizes = context.get_input(0);
    auto value = context.mark_node(opset8::Constant::create(element::f32, Shape{}, {0}));
    auto filled_tensor = context.mark_node(std::make_shared<opset8::Broadcast>(value, sizes));
    int num_inputs = context.get_input_size();
    if (num_inputs < 5) {
        size_t out_id = num_inputs == 2 ? 1: 2;
        if (!context.input_is_none(out_id)){
        auto out = context.get_input(out_id);
        return {context.mark_node(std::make_shared<opset8::ConvertLike>(filled_tensor, out))};
        }
        return {filled_tensor};
    }
    size_t dtype_id = num_inputs == 5 ? 1: 2;
    if (!context.input_is_none(dtype_id)){
        std::cout << dtype_id << std::endl;
        auto pt_type = context.const_input<int64_t>(dtype_id);
        std::cout << pt_type << std::endl;
        FRONT_END_OP_CONVERSION_CHECK(TORCH_TO_OV_TYPE.count(pt_type), "Unknown type in aten::zeros: ", pt_type);
        auto dtype = TORCH_TO_OV_TYPE.at(pt_type);
        filled_tensor = context.mark_node(std::make_shared<opset8::Convert>(filled_tensor, dtype));
    }
    return {filled_tensor};
};

OutputVector translate_zeros_like(NodeContext& context) {
    auto input = context.get_input(0);
    auto value = context.mark_node(opset8::Constant::create(element::f32, Shape{}, {0}));
    auto input_shape = context.mark_node(std::make_shared<opset8::ShapeOf>(input));
    auto filled_tensor = context.mark_node(std::make_shared<opset8::Broadcast>(value, input_shape));
    if (context.get_input_size() == 6 && !context.input_is_none(1)){
        auto pt_type = context.const_input<int64_t>(1);
        FRONT_END_OP_CONVERSION_CHECK(TORCH_TO_OV_TYPE.count(pt_type), "Unknown type in aten::zeros_like: ", pt_type);
        auto dtype = TORCH_TO_OV_TYPE.at(pt_type);
        filled_tensor = context.mark_node(std::make_shared<opset8::Convert>(filled_tensor, dtype));
    }
    else {
        auto out_dtype = context.input_is_none(2)? input : context.get_input(2);
        filled_tensor = context.mark_node(std::make_shared<opset8::ConvertLike>(filled_tensor, out_dtype));
    }
    return {filled_tensor};
};

OutputVector translate_new_zeros(NodeContext& context) {
    auto input = context.get_input(0);
    auto sizes = context.get_input(1);
    auto value = context.mark_node(opset8::Constant::create(element::f32, Shape{}, {0}));
    auto filled_tensor = context.mark_node(std::make_shared<opset8::Broadcast>(value, sizes));
    if (context.get_input_size() == 6 && !context.input_is_none(2)){
        auto pt_type = context.const_input<int64_t>(2);
        FRONT_END_OP_CONVERSION_CHECK(TORCH_TO_OV_TYPE.count(pt_type), "Unknown type in aten::new_zeros: ", pt_type);
        auto dtype = TORCH_TO_OV_TYPE.at(pt_type);
        return {context.mark_node(std::make_shared<opset8::Convert>(filled_tensor, dtype))};
    }
    return {context.mark_node(std::make_shared<opset8::ConvertLike>(filled_tensor, input))};
};

OutputVector translate_ones(NodeContext& context) {
    auto sizes = context.get_input(0);
    auto value = context.mark_node(opset8::Constant::create(element::f32, Shape{}, {1}));
    auto filled_tensor = context.mark_node(std::make_shared<opset8::Broadcast>(value, sizes));
    int num_inputs = context.get_input_size();
    if (num_inputs < 5) {
        size_t out_id = num_inputs == 2 ? 1: 2;
        if (!context.input_is_none(out_id)){
        auto out = context.get_input(out_id);
        return {context.mark_node(std::make_shared<opset8::ConvertLike>(filled_tensor, out))};
        }
    }
    size_t dtype_id = num_inputs == 5 ? 1: 2;
    if (!context.input_is_none(dtype_id)){
        auto pt_type = context.const_input<int64_t>(dtype_id);
        FRONT_END_OP_CONVERSION_CHECK(TORCH_TO_OV_TYPE.count(pt_type), "Unknown type in aten::ones: ", pt_type);
        auto dtype = TORCH_TO_OV_TYPE.at(pt_type);
        filled_tensor = context.mark_node(std::make_shared<opset8::Convert>(filled_tensor, dtype));
    }
    return {filled_tensor};
};

OutputVector translate_ones_like(NodeContext& context) {
    auto input = context.get_input(0);
    auto value = context.mark_node(opset8::Constant::create(element::f32, Shape{}, {1}));
    auto input_shape = context.mark_node(std::make_shared<opset8::ShapeOf>(input));
    auto filled_tensor = context.mark_node(std::make_shared<opset8::Broadcast>(value, input_shape));
    if (context.get_input_size() == 6 && !context.input_is_none(1)){
        auto pt_type = context.const_input<int64_t>(1);
        FRONT_END_OP_CONVERSION_CHECK(TORCH_TO_OV_TYPE.count(pt_type), "Unknown type in aten::ones_like: ", pt_type);
        auto dtype = TORCH_TO_OV_TYPE.at(pt_type);
        filled_tensor = context.mark_node(std::make_shared<opset8::Convert>(filled_tensor, dtype));
    }
    else {
        auto out_dtype = context.input_is_none(2)? input : context.get_input(2);
        filled_tensor = context.mark_node(std::make_shared<opset8::ConvertLike>(filled_tensor, out_dtype));
    }
    return {filled_tensor};
};

OutputVector translate_new_ones(NodeContext& context) {
    auto input = context.get_input(0);
    auto sizes = context.get_input(1);
    auto value = context.mark_node(opset8::Constant::create(element::f32, Shape{}, {1}));
    auto filled_tensor = context.mark_node(std::make_shared<opset8::Broadcast>(value, sizes));
    if (context.get_input_size() == 6 && !context.input_is_none(2)){
        auto pt_type = context.const_input<int64_t>(2);
        FRONT_END_OP_CONVERSION_CHECK(TORCH_TO_OV_TYPE.count(pt_type), "Unknown type in aten::new_zeros: ", pt_type);
        auto dtype = TORCH_TO_OV_TYPE.at(pt_type);
        return {context.mark_node(std::make_shared<opset8::Convert>(filled_tensor, dtype))};
    }
    return {context.mark_node(std::make_shared<opset8::ConvertLike>(filled_tensor, input))};
};

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov