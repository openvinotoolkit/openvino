// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "op_table.hpp"

#include "openvino/opsets/opset8.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

#define OP_CONVERTER(op) OutputVector op(NodeContext& node)

OP_CONVERTER(translate_adaptive_max_pool2d);
OP_CONVERTER(translate_add);
OP_CONVERTER(translate_addcmul);
OP_CONVERTER(translate_as_tensor);
OP_CONVERTER(translate_avg_pool2d);
OP_CONVERTER(translate_batch_norm);
OP_CONVERTER(translate_constant);
OP_CONVERTER(translate_conv2d);
OP_CONVERTER(translate_convolution);
OP_CONVERTER(translate_dim);
OP_CONVERTER(translate_div);
OP_CONVERTER(translate_elu);
OP_CONVERTER(translate_expand);
OP_CONVERTER(translate_expand_as);
OP_CONVERTER(translate_embedding);
OP_CONVERTER(translate_flatten);
OP_CONVERTER(translate_floordiv);
OP_CONVERTER(translate_gelu);
OP_CONVERTER(translate_get_attr);
OP_CONVERTER(translate_group_norm);
OP_CONVERTER(translate_hardtanh);
OP_CONVERTER(translate_if);
OP_CONVERTER(translate_int);
OP_CONVERTER(translate_layer_norm);
OP_CONVERTER(translate_linear);
OP_CONVERTER(translate_list_construct);
OP_CONVERTER(translate_loop);
OP_CONVERTER(translate_max_pool2d);
OP_CONVERTER(translate_mean);
OP_CONVERTER(translate_neg);
OP_CONVERTER(translate_reciprocal);
OP_CONVERTER(translate_relu6);
OP_CONVERTER(translate_reshape);
OP_CONVERTER(translate_reshape_as);
OP_CONVERTER(translate_rsub);
OP_CONVERTER(translate_select);
OP_CONVERTER(translate_size);
OP_CONVERTER(translate_slice);
OP_CONVERTER(translate_softmax);
OP_CONVERTER(translate_square);
OP_CONVERTER(translate_squeeze);
OP_CONVERTER(translate_sub);
OP_CONVERTER(translate_sum);
OP_CONVERTER(translate_to);
OP_CONVERTER(translate_transpose);
OP_CONVERTER(translate_tuple_construct);
OP_CONVERTER(translate_upsample_bilinear2d);
OP_CONVERTER(translate_upsample_nearest2d);
OP_CONVERTER(translate_view);

}  // namespace op

const std::map<std::string, CreatorFunction> get_supported_ops() {
    return {
        {"aten::_convolution", op::translate_convolution},
        {"aten::abs", op::translate_1to1_match_1_inputs<opset8::Abs>},
        {"aten::adaptive_avg_pool2d", op::translate_1to1_match_2_inputs<opset8::AdaptiveAvgPool>},
        {"aten::adaptive_max_pool2d", op::translate_adaptive_max_pool2d},
        {"aten::add", op::translate_add},
        {"aten::add_", op::inplace_op<op::translate_add>},
        {"aten::addcmul", op::translate_addcmul},
        {"aten::as_tensor", op::translate_as_tensor},
        {"aten::avg_pool2d", op::translate_avg_pool2d},
        {"aten::batch_norm", op::translate_batch_norm},
        // {"aten::cat", done as transformation},
        {"aten::contiguous", op::skip_node},  // In openvino how tensors are stored in memory is internal plugin detail,
                                              // we assume all tensors are contiguous
        {"aten::conv2d", op::translate_conv2d},
        {"aten::dim", op::translate_dim},
        {"aten::div", op::translate_div},
        {"aten::div_", op::inplace_op<op::translate_div>},
        {"aten::dropout", op::skip_node},
        {"aten::dropout_", op::skip_node},
        {"aten::elu", op::translate_elu},
        {"aten::embedding", op::translate_embedding},
        {"aten::eq", op::translate_1to1_match_2_inputs<opset8::Equal>},
        {"aten::exp", op::translate_1to1_match_1_inputs<opset8::Exp>},
        {"aten::expand", op::translate_expand},
        {"aten::expand_as", op::translate_expand_as},
        {"aten::flatten", op::translate_flatten},
        {"aten::floordiv", op::translate_floordiv},
        {"aten::gelu", op::translate_gelu},
        {"aten::group_norm", op::translate_group_norm},
        {"aten::gt", op::translate_1to1_match_2_inputs<opset8::Greater>},
        {"aten::hardsigmoid", op::translate_1to1_match_1_inputs<opset8::HSigmoid>},
        {"aten::hardswish", op::translate_1to1_match_1_inputs<opset8::HSwish>},
        {"aten::hardswish_", op::inplace_op<op::translate_1to1_match_1_inputs<opset8::HSwish>>},
        {"aten::hardtanh", op::translate_hardtanh},
        {"aten::hardtanh_", op::inplace_op<op::translate_hardtanh>},
        {"aten::Int", op::translate_int},
        {"aten::is_grad_enabled", op::return_false_scalar},
        {"aten::layer_norm", op::translate_layer_norm},
        {"aten::leaky_relu", op::translate_1to1_match_2_inputs<opset8::PRelu>},
        {"aten::leaky_relu_", op::inplace_op<op::translate_1to1_match_2_inputs<opset8::PRelu>>},
        {"aten::linear", op::translate_linear},
        {"aten::lt", op::translate_1to1_match_2_inputs<opset8::Less>},
        {"aten::matmul", op::translate_1to1_match_2_inputs<opset8::MatMul>},
        {"aten::max_pool2d", op::translate_max_pool2d},
        {"aten::mean", op::translate_mean},
        {"aten::mm", op::translate_1to1_match_2_inputs<opset8::MatMul>},
        {"aten::mul", op::translate_1to1_match_2_inputs<opset8::Multiply>},
        {"aten::mul_", op::inplace_op<op::translate_1to1_match_2_inputs<opset8::Multiply>>},
        {"aten::ne", op::translate_1to1_match_2_inputs<opset8::NotEqual>},
        {"aten::neg", op::translate_neg},
        {"aten::permute", op::translate_1to1_match_2_inputs<opset8::Transpose>},
        {"aten::pow", op::translate_1to1_match_2_inputs<opset8::Power>},
        {"aten::reciprocal", op::translate_reciprocal},
        {"aten::relu", op::translate_1to1_match_1_inputs<opset8::Relu>},
        {"aten::relu_", op::inplace_op<op::translate_1to1_match_1_inputs<opset8::Relu>>},
        {"aten::relu6", op::translate_relu6},
        {"aten::reshape", op::translate_reshape},
        {"aten::reshape_as", op::translate_reshape_as},
        {"aten::rsub", op::translate_rsub},
        {"aten::select", op::translate_select},
        {"aten::sigmoid", op::translate_1to1_match_1_inputs<opset8::Sigmoid>},
        {"aten::silu", op::translate_1to1_match_1_inputs<opset8::Swish>},
        {"aten::silu_", op::inplace_op<op::translate_1to1_match_1_inputs<opset8::Swish>>},
        {"aten::size", op::translate_size},
        {"aten::slice", op::translate_slice},
        {"aten::softmax", op::translate_softmax},
        {"aten::sqrt", op::translate_1to1_match_1_inputs<opset8::Sqrt>},
        {"aten::square", op::translate_square},
        {"aten::squeeze", op::translate_squeeze},
        {"aten::sub", op::translate_sub},
        {"aten::sum", op::translate_sum},
        {"aten::tanh", op::translate_1to1_match_1_inputs<opset8::Tanh>},
        {"aten::to", op::translate_to},
        {"aten::transpose", op::translate_transpose},
        {"aten::unsqueeze", op::translate_1to1_match_2_inputs<opset8::Unsqueeze>},
        {"aten::upsample_bilinear2d", op::translate_upsample_bilinear2d},
        {"aten::upsample_nearest2d", op::translate_upsample_nearest2d},
        {"aten::view", op::translate_view},
        {"prim::Constant", op::translate_constant},
        {"prim::GetAttr", op::translate_get_attr},
        {"prim::If", op::translate_if},
        {"prim::is_cuda", op::return_false_scalar},
        {"prim::ListConstruct", op::translate_list_construct},
        {"prim::Loop", op::translate_loop},
        {"prim::NumToTensor", op::skip_node},  // In openvino we already store number as tensor with shape []
        {"prim::requires_grad", op::return_false_scalar},
        {"prim::TupleConstruct", op::translate_tuple_construct},
    };
};

}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
