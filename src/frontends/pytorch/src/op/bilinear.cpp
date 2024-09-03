// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/reduce_sum.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

OutputVector translate_bilinear(const NodeContext& context) {
    // schema: aten::bilinear(Tensor input1, Tensor input2, Tensor weight, Tensor? bias=None) -> Tensor
    num_inputs_check(context, 3, 4);
    auto x0 = context.get_input(0);
    auto x1 = context.get_input(1);
    auto weight = context.get_input(2);
    if (weight.get_element_type() == element::f16 || weight.get_element_type() == element::bf16) {
        // In case of patched bilinear it can have mixed fp16/bf16 and fp32 input type.
        // In other cases these conversion is not required.
        weight = context.mark_node(std::make_shared<ov::op::v1::ConvertLike>(weight, x0));
    }
    auto un_squeeze_axis0 = context.mark_node(
        std::make_shared<ov::op::v0::Constant>(element::i32, Shape{2}, std::vector<int32_t>({1,3})));    
    auto un_squeeze_axis1 = context.mark_node(
        std::make_shared<ov::op::v0::Constant>(element::i32, Shape{1}, std::vector<int32_t>({0})));    
    auto un_squeeze_axis2 = context.mark_node(
        std::make_shared<ov::op::v0::Constant>(element::i32, Shape{2}, std::vector<int32_t>({1,2})));    
    auto un_squeeze_axis3 = context.mark_node(
        std::make_shared<ov::op::v0::Constant>(element::i32, Shape{2}, std::vector<int32_t>({2, 3})));    

    x0 = context.mark_node(std::make_shared<ov::op::v0::Unsqueeze>(x0, un_squeeze_axis0));

    weight = context.mark_node(std::make_shared<ov::op::v0::Unsqueeze>(weight, un_squeeze_axis1));

    x1 = context.mark_node(std::make_shared<ov::op::v0::Unsqueeze>(x1, un_squeeze_axis2));

    std::cout << "### x0=" << x0.get_partial_shape() << ", weight=" << weight.get_partial_shape()  << ", x1=" << x1.get_partial_shape() << std::endl;


    auto multiply = context.mark_node(std::make_shared<ov::op::v1::Multiply>(x0, weight));

    std::cout << "### multiply1=" << multiply->get_output_partial_shape(0) << std::endl;
    multiply = context.mark_node(std::make_shared<ov::op::v1::Multiply>(multiply, x1));
    std::cout << "### multiply2=" << multiply->get_output_partial_shape(0) << std::endl;
    multiply = context.mark_node(std::make_shared<ov::op::v1::ReduceSum>(multiply, un_squeeze_axis3));
    std::cout << "### ReduceSum=" << multiply->get_output_partial_shape(0) << std::endl;

    
    if (!context.input_is_none(3)) {
        auto bias = context.get_input(3);

        if (bias.get_element_type() == element::f16 || bias.get_element_type() == element::bf16) {
            // Same reason as for weight.
            bias = context.mark_node(std::make_shared<ov::op::v1::ConvertLike>(bias, x0));
        }
        multiply = context.mark_node(std::make_shared<ov::op::v1::Add>(multiply, bias));
    }
    return {multiply};
};


}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov