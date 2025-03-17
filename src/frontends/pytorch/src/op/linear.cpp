// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/subtract.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

using namespace ov::op;

OutputVector translate_linear(const NodeContext& context) {
    // schema: aten::linear(Tensor input, Tensor weight, Tensor? bias=None) -> Tensor
    num_inputs_check(context, 2, 3);
    auto x = context.get_input(0);
    auto weight = context.get_input(1);
    auto matmul = context.mark_node(std::make_shared<v0::MatMul>(x, weight, false, true));
    if (!context.input_is_none(2)) {
        auto bias = context.get_input(2);
        matmul = context.mark_node(std::make_shared<v1::Add>(matmul, bias));
    }
    return {matmul};
};

OutputVector translate_linear_ext(const NodeContext& context) {
    num_inputs_check(context, 2, 3);
    auto x = context.get_input(0);
    auto initial_x = x;
    auto weight = context.get_input(1);
    bool convert_back = false;
    if (weight.get_element_type() != element::f32) {
        // In case of patched linear it can have mixed fp16/bf16 and fp32 input type.
        // In other cases these conversion is not required.
        weight = context.mark_node(std::make_shared<v0::Convert>(weight, element::f32));
        if (x.get_element_type() != element::f32) {
            // Convert to f32
            x = context.mark_node(std::make_shared<v0::Convert>(x, element::f32));
            convert_back = true;
        }
    }
    auto matmul = context.mark_node(std::make_shared<v0::MatMul>(x, weight, false, true));
    if (!context.input_is_none(2)) {
        auto bias = context.get_input(2);

        if (bias.get_element_type() != element::f32) {
            // Same reason as for weight.
            bias = context.mark_node(std::make_shared<v0::Convert>(bias, element::f32));
        }
        matmul = context.mark_node(std::make_shared<v1::Add>(matmul, bias));
    }
    if (convert_back) {
        matmul = context.mark_node(std::make_shared<v1::ConvertLike>(matmul, initial_x));
    }
    return {matmul};
};

namespace {
uint32_t rearrange_awq_bits(uint32_t num) {
    uint32_t result = 0;
    uint32_t mask = 0xF;

    // Rearrange each 4-bit part in accordance with the AWQ i32->u4 unpacking schema
    result |= (num & (mask << 0)) << 0;
    result |= (num & (mask << 16)) >> 12;
    result |= (num & (mask << 4)) << 4;
    result |= (num & (mask << 20)) >> 8;
    result |= (num & (mask << 8)) << 8;
    result |= (num & (mask << 24)) >> 4;
    result |= (num & (mask << 12)) << 12;
    result |= (num & (mask << 28)) >> 0;

    return result;
}

Output<Node> rearrange_constant(const Output<Node>& c, uint32_t groups) {
    auto constant = ov::as_type_ptr<v0::Constant>(c.get_node_shared_ptr());
    FRONT_END_OP_CONVERSION_CHECK(constant, "weight must be Constant.");
    auto src = constant->get_data_ptr<uint32_t>();
    auto initial_shape = constant->get_shape();
    FRONT_END_OP_CONVERSION_CHECK(initial_shape.size() == 2, "Only 2D constants are supported.");
    auto new_shape = Shape{initial_shape[0] / groups, groups, initial_shape[1] * 8};
    auto new_qweight = std::make_shared<v0::Constant>(element::u4, new_shape);
    auto dst = const_cast<uint32_t*>(reinterpret_cast<const uint32_t*>(new_qweight->get_data_ptr()));
    for (size_t i = 0; i < shape_size(constant->get_shape()); i++) {
        dst[i] = rearrange_awq_bits(src[i]);
    }
    return new_qweight;
}
}  // namespace

OutputVector translate_linear_awq(const NodeContext& context) {
    num_inputs_check(context, 4, 7);
    auto x = context.get_input(0);
    auto qweight = context.get_input(1);
    auto qzeros = context.get_input(2);
    auto scales = context.get_input(3);
    auto groups = context.const_input<int64_t>(4);
    auto bits = context.const_input<int64_t>(5);

    FRONT_END_OP_CONVERSION_CHECK(bits == 4, "Only 4 bit AWQ is supported.");

    auto new_qweight = rearrange_constant(qweight, static_cast<uint32_t>(groups));
    auto new_qzeros = rearrange_constant(qzeros, 1);
    new_qweight = context.mark_node(std::make_shared<v0::Convert>(new_qweight, scales.get_element_type()));
    new_qzeros = context.mark_node(std::make_shared<v0::Convert>(new_qzeros, scales.get_element_type()));

    auto w_s = context.mark_node(std::make_shared<v1::Subtract>(new_qweight, new_qzeros));
    FRONT_END_OP_CONVERSION_CHECK(scales.get_partial_shape().is_static(), "Scales must be constant.");
    auto scales_shape = scales.get_shape();
    auto new_scales_shape =
        v0::Constant::create(element::i32, {3}, std::vector<uint64_t>{scales_shape[0], 1, scales_shape[1]});
    scales = context.mark_node(std::make_shared<v1::Reshape>(scales, new_scales_shape, false));
    auto weight = context.mark_node(std::make_shared<v1::Multiply>(w_s, scales));
    auto out_shape =
        v0::Constant::create(element::i32, {2}, std::vector<int32_t>{static_cast<int32_t>(qweight.get_shape()[0]), -1});
    weight = context.mark_node(std::make_shared<v1::Reshape>(weight, out_shape, false));
    weight = context.mark_node(std::make_shared<v1::ConvertLike>(weight, x));

    auto matmul = context.mark_node(std::make_shared<v0::MatMul>(x, weight, false, false));
    if (!context.input_is_none(6)) {
        auto bias = context.get_input(6);

        if (bias.get_element_type() == element::f16 || bias.get_element_type() == element::bf16) {
            bias = context.mark_node(std::make_shared<v1::ConvertLike>(bias, x));
        }
        matmul = context.mark_node(std::make_shared<v1::Add>(matmul, bias));
    }
    return {matmul};
};

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
