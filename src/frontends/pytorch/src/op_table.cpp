// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "op_table.hpp"

#include "openvino/opsets/opset10.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

#define OP_CONVERTER(op) OutputVector op(const NodeContext& node)


//TorchScript translations
OP_CONVERTER(translate_adaptive_avg_pool3d);
OP_CONVERTER(translate_adaptive_max_pool2d);
OP_CONVERTER(translate_add);
OP_CONVERTER(translate_addcmul);
OP_CONVERTER(translate_addmm);
OP_CONVERTER(translate_all);
OP_CONVERTER(translate_arange);
OP_CONVERTER(translate_argsort);
OP_CONVERTER(translate_argmax);
OP_CONVERTER(translate_argmin);
OP_CONVERTER(translate_as_tensor);
OP_CONVERTER(translate_avg_poolnd);
OP_CONVERTER(translate_bool);
OP_CONVERTER(translate_batch_norm);
OP_CONVERTER(translate_bitwise_and);
OP_CONVERTER(translate_bitwise_not);
OP_CONVERTER(translate_bitwise_or);
OP_CONVERTER(translate_cat);
OP_CONVERTER(translate_chunk);
OP_CONVERTER(translate_clamp);
OP_CONVERTER(translate_constant);
OP_CONVERTER(translate_conv_transposend);
OP_CONVERTER(translate_convnd);
OP_CONVERTER(translate_convolution);
OP_CONVERTER(translate_convolution_mode);
OP_CONVERTER(translate_cumsum);
OP_CONVERTER(translate_deform_conv);
OP_CONVERTER(translate_dim);
OP_CONVERTER(translate_div);
OP_CONVERTER(translate_elu);
OP_CONVERTER(translate_embedding);
OP_CONVERTER(translate_embedding_bag);
OP_CONVERTER(translate_empty);
OP_CONVERTER(translate_expand);
OP_CONVERTER(translate_expand_as);
OP_CONVERTER(translate_eye);
OP_CONVERTER(translate_fill_);
OP_CONVERTER(translate_flatten);
OP_CONVERTER(translate_floor_divide);
OP_CONVERTER(translate_floordiv);
OP_CONVERTER(translate_full);
OP_CONVERTER(translate_full_like);
OP_CONVERTER(translate_gather);
OP_CONVERTER(translate_gelu);
OP_CONVERTER(translate_get_attr);
OP_CONVERTER(translate_getitem);
OP_CONVERTER(translate_glu);
OP_CONVERTER(translate_grid_sampler);
OP_CONVERTER(translate_group_norm);
OP_CONVERTER(translate_hardtanh);
OP_CONVERTER(translate_if);
OP_CONVERTER(translate_im2col);
OP_CONVERTER(translate_index_put_);
OP_CONVERTER(translate_index_select);
OP_CONVERTER(translate_instance_norm);
OP_CONVERTER(translate_int);
OP_CONVERTER(translate_layer_norm);
OP_CONVERTER(translate_len);
OP_CONVERTER(translate_linalg_norm);
OP_CONVERTER(translate_linalg_matrix_norm);
OP_CONVERTER(translate_linalg_vector_norm);
OP_CONVERTER(translate_linear);
OP_CONVERTER(translate_list_construct);
OP_CONVERTER(translate_log);
OP_CONVERTER(translate_log2);
OP_CONVERTER(translate_logsumexp);
OP_CONVERTER(translate_loop);
OP_CONVERTER(translate_masked_fill);
OP_CONVERTER(translate_max);
OP_CONVERTER(translate_max_poolnd);
OP_CONVERTER(translate_mean);
OP_CONVERTER(translate_meshgrid);
OP_CONVERTER(translate_min);
OP_CONVERTER(translate_narrow);
OP_CONVERTER(translate_neg);
OP_CONVERTER(translate_new_full);
OP_CONVERTER(translate_new_ones);
OP_CONVERTER(translate_new_zeros);
OP_CONVERTER(translate_nms);
OP_CONVERTER(translate_nonzero);
OP_CONVERTER(translate_norm);
OP_CONVERTER(translate_numel);
OP_CONVERTER(translate_ones);
OP_CONVERTER(translate_ones_like);
OP_CONVERTER(translate_pad);
OP_CONVERTER(translate_pow);
OP_CONVERTER(translate_pythonop);
OP_CONVERTER(translate_reciprocal);
OP_CONVERTER(translate_relu6);
OP_CONVERTER(translate_remainder);
OP_CONVERTER(translate_repeat);
OP_CONVERTER(translate_repeat_interleave);
OP_CONVERTER(translate_reshape);
OP_CONVERTER(translate_reshape_as);
OP_CONVERTER(translate_roi_align);
OP_CONVERTER(translate_roll);
OP_CONVERTER(translate_rsqrt);
OP_CONVERTER(translate_rsub);
OP_CONVERTER(translate_scaled_dot_product_attention);
OP_CONVERTER(translate_select);
OP_CONVERTER(translate_set_item);
OP_CONVERTER(translate_selu);
OP_CONVERTER(translate_sign);
OP_CONVERTER(translate_size);
OP_CONVERTER(translate_slice);
OP_CONVERTER(translate_softmax);
OP_CONVERTER(translate_sort);
OP_CONVERTER(translate_square);
OP_CONVERTER(translate_squeeze);
OP_CONVERTER(translate_sub);
OP_CONVERTER(translate_sum);
OP_CONVERTER(translate_to);
OP_CONVERTER(translate_topk);
OP_CONVERTER(translate_transpose);
OP_CONVERTER(translate_tril);
OP_CONVERTER(translate_triu);
OP_CONVERTER(translate_unfold);
OP_CONVERTER(translate_upsample_bicubic2d);
OP_CONVERTER(translate_upsample_bilinear2d);
OP_CONVERTER(translate_upsample_linear1d);
OP_CONVERTER(translate_upsample_nearest1d);
OP_CONVERTER(translate_upsample_nearest2d);
OP_CONVERTER(translate_upsample_nearest3d);
OP_CONVERTER(translate_upsample_trilinear3d);
OP_CONVERTER(translate_var);
OP_CONVERTER(translate_var_mean);
OP_CONVERTER(translate_where);
OP_CONVERTER(translate_zeros);
OP_CONVERTER(translate_zeros_like);
//Torch FX Translations
OP_CONVERTER(translate_batch_norm_fx);
OP_CONVERTER(translate_cat_fx);
OP_CONVERTER(translate_chunk_fx);
OP_CONVERTER(translate_group_norm_fx);
OP_CONVERTER(translate_max_poolnd_fx);
OP_CONVERTER(translate_slice_fx);
OP_CONVERTER(translate_softmax_fx);
OP_CONVERTER(translate_transpose_fx);
}  // namespace op


//Supported ops for TorchScript
const std::map<std::string, CreatorFunction> get_supported_ops_ts() {
    return {
        {"aten::__and__", op::translate_1to1_match_2_inputs<opset10::LogicalAnd>},  // TODO: cover numerical cases
        {"aten::__getitem__", op::translate_getitem},
        {"aten::__not__", op::translate_1to1_match_1_inputs<opset10::LogicalNot>},
        {"aten::__or__", op::translate_1to1_match_2_inputs<opset10::LogicalOr>},
        {"aten::_convolution", op::translate_convolution},
        {"aten::_convolution_mode", op::translate_convolution_mode},
        {"aten::_set_item", op::translate_set_item},
        {"aten::abs", op::translate_1to1_match_1_inputs<opset10::Abs>},
        {"aten::acos", op::translate_1to1_match_1_inputs_with_fp32_type_alignment<opset10::Acos>},
        {"aten::acos_", op::inplace_op<op::translate_1to1_match_1_inputs<opset10::Acos>>},
        {"aten::acosh", op::translate_1to1_match_1_inputs_with_fp32_type_alignment<opset10::Acosh>},
        {"aten::acosh_", op::inplace_op<op::translate_1to1_match_1_inputs<opset10::Acosh>>},
        {"aten::adaptive_avg_pool2d", op::translate_1to1_match_2_inputs<opset10::AdaptiveAvgPool>},
        {"aten::adaptive_avg_pool3d", op::translate_adaptive_avg_pool3d},
        {"aten::adaptive_max_pool2d", op::translate_adaptive_max_pool2d},
        {"aten::add", op::translate_add},
        {"aten::add_", op::inplace_op<op::translate_add>},
        {"aten::addcmul", op::translate_addcmul},
        {"aten::addmm", op::translate_addmm},
        {"aten::all", op::translate_all},
        {"aten::arange", op::translate_arange},
        {"aten::argmax", op::translate_argmax},
        {"aten::argmin", op::translate_argmin},
        {"aten::argsort", op::translate_argsort},
        {"aten::as_tensor", op::translate_as_tensor},
        {"aten::asin", op::translate_1to1_match_1_inputs_with_fp32_type_alignment<opset10::Asin>},
        {"aten::asin_", op::inplace_op<op::translate_1to1_match_1_inputs<opset10::Asin>>},
        {"aten::asinh", op::translate_1to1_match_1_inputs_with_fp32_type_alignment<opset10::Asinh>},
        {"aten::asinh_", op::inplace_op<op::translate_1to1_match_1_inputs<opset10::Asinh>>},
        {"aten::atan", op::translate_1to1_match_1_inputs_with_fp32_type_alignment<opset10::Atan>},
        {"aten::atan_", op::inplace_op<op::translate_1to1_match_1_inputs<opset10::Atan>>},
        {"aten::atanh", op::translate_1to1_match_1_inputs_with_fp32_type_alignment<opset10::Atanh>},
        {"aten::atanh_", op::inplace_op<op::translate_1to1_match_1_inputs<opset10::Atanh>>},
        {"aten::avg_pool1d", op::translate_avg_poolnd},
        {"aten::avg_pool2d", op::translate_avg_poolnd},
        {"aten::avg_pool3d", op::translate_avg_poolnd},
        {"aten::baddbmm", op::translate_addmm},
        {"aten::batch_norm", op::translate_batch_norm},
        {"aten::bitwise_not", op::translate_bitwise_not},
        {"aten::bmm", op::translate_1to1_match_2_inputs<opset10::MatMul>},
        {"aten::Bool", op::translate_bool},
        {"aten::cat", op::translate_cat},
        {"aten::ceil", op::translate_1to1_match_1_inputs<opset10::Ceiling>},
        {"aten::ceil_", op::inplace_op<op::translate_1to1_match_1_inputs<opset10::Ceiling>>},
        {"aten::chunk", op::translate_chunk},
        {"aten::clamp", op::translate_clamp},
        {"aten::clamp_max", op::translate_1to1_match_2_inputs<opset10::Minimum>},
        {"aten::clamp_min", op::translate_1to1_match_2_inputs<opset10::Maximum>},
        {"aten::clone", op::skip_node},       // ignore clone operators that are inserted by PyTorch autograd
        {"aten::contiguous", op::skip_node},  // In openvino how tensors are stored in memory is internal plugin detail,
                                              // we assume all tensors are contiguous
        {"aten::conv_transpose1d", op::translate_conv_transposend},
        {"aten::conv_transpose2d", op::translate_conv_transposend},
        {"aten::conv_transpose3d", op::translate_conv_transposend},
        {"aten::conv1d", op::translate_convnd},
        {"aten::conv2d", op::translate_convnd},
        {"aten::conv3d", op::translate_convnd},
        {"aten::convolution", op::translate_convolution},
        {"aten::copy", op::skip_node},
        {"aten::cos", op::translate_1to1_match_1_inputs_with_fp32_type_alignment<opset10::Cos>},
        {"aten::cos_", op::inplace_op<op::translate_1to1_match_1_inputs<opset10::Cos>>},
        {"aten::cosh", op::translate_1to1_match_1_inputs_with_fp32_type_alignment<opset10::Cosh>},
        {"aten::cosh_", op::inplace_op<op::translate_1to1_match_1_inputs<opset10::Cosh>>},
        {"aten::cumsum", op::translate_cumsum},
        {"aten::detach", op::skip_node},
        {"aten::dim", op::translate_dim},
        {"aten::div", op::translate_div},
        {"aten::div_", op::inplace_op<op::translate_div>},
        {"aten::dropout", op::skip_node},
        {"aten::dropout_", op::skip_node},
        {"aten::elu", op::translate_elu},
        {"aten::embedding", op::translate_embedding},
        {"aten::embedding_bag", op::translate_embedding_bag},
        {"aten::empty", op::translate_empty},
        {"aten::eq", op::translate_1to1_match_2_inputs_align_types<opset10::Equal>},
        {"aten::exp", op::translate_1to1_match_1_inputs_with_fp32_type_alignment<opset10::Exp>},
        {"aten::expand", op::translate_expand},
        {"aten::expand_as", op::translate_expand_as},
        {"aten::eye", op::translate_eye},
        {"aten::fill_", op::inplace_op<op::translate_fill_>},
        {"aten::flatten", op::translate_flatten},
        {"aten::floor", op::translate_1to1_match_1_inputs<opset10::Floor>},
        {"aten::floor_", op::inplace_op<op::translate_1to1_match_1_inputs<opset10::Floor>>},
        {"aten::floor_divide", op::translate_floor_divide},
        {"aten::floordiv", op::translate_floordiv},
        {"aten::full", op::translate_full},
        {"aten::full_like", op::translate_full_like},
        {"aten::gather", op::translate_gather},
        {"aten::ge", op::translate_1to1_match_2_inputs_align_types<opset10::GreaterEqual>},
        {"aten::gelu", op::translate_gelu},
        {"aten::glu", op::translate_glu},
        {"aten::grid_sampler", op::translate_grid_sampler},
        {"aten::group_norm", op::translate_group_norm},
        {"aten::gt", op::translate_1to1_match_2_inputs_align_types<opset10::Greater>},
        {"aten::hardsigmoid", op::translate_1to1_match_1_inputs<opset10::HSigmoid>},
        {"aten::hardswish", op::translate_1to1_match_1_inputs<opset10::HSwish>},
        {"aten::hardswish_", op::inplace_op<op::translate_1to1_match_1_inputs<opset10::HSwish>>},
        {"aten::hardtanh", op::translate_hardtanh},
        {"aten::hardtanh_", op::inplace_op<op::translate_hardtanh>},
        {"aten::im2col", op::translate_im2col},
        {"aten::index_put_", op::inplace_op<op::translate_index_put_>},
        {"aten::index_select", op::translate_index_select},
        {"aten::instance_norm", op::translate_instance_norm},
        {"aten::Int", op::translate_int},
        {"aten::IntImplicit", op::translate_int},
        {"aten::is_grad_enabled", op::return_false_scalar},
        {"aten::layer_norm", op::translate_layer_norm},
        {"aten::le", op::translate_1to1_match_2_inputs_align_types<opset10::LessEqual>},
        {"aten::leaky_relu", op::translate_1to1_match_2_inputs<opset10::PRelu>},
        {"aten::leaky_relu_", op::inplace_op<op::translate_1to1_match_2_inputs<opset10::PRelu>>},
        {"aten::len", op::translate_len},
        {"aten::linalg_norm", op::translate_linalg_norm},
        {"aten::linalg_matrix_norm", op::translate_linalg_matrix_norm},
        {"aten::linalg_vector_norm", op::translate_linalg_vector_norm},
        {"aten::linear", op::translate_linear},
        {"aten::log", op::translate_log},
        {"aten::log_", op::inplace_op<op::translate_log>},
        {"aten::log2", op::translate_log2},
        {"aten::log2_", op::inplace_op<op::translate_log2>},
        {"aten::lt", op::translate_1to1_match_2_inputs_align_types<opset10::Less>},
        {"aten::masked_fill", op::translate_masked_fill},
        {"aten::masked_fill_", op::inplace_op<op::translate_masked_fill>},
        {"aten::matmul", op::translate_1to1_match_2_inputs<opset10::MatMul>},
        {"aten::max", op::translate_max},
        {"aten::max_pool1d", op::translate_max_poolnd},
        {"aten::max_pool2d", op::translate_max_poolnd},
        {"aten::max_pool3d", op::translate_max_poolnd},
        {"aten::mean", op::translate_mean},
        {"aten::meshgrid", op::translate_meshgrid},
        {"aten::min", op::translate_min},
        {"aten::mm", op::translate_1to1_match_2_inputs<opset10::MatMul>},
        {"aten::mul", op::translate_1to1_match_2_inputs_align_types<opset10::Multiply>},
        {"aten::mul_", op::inplace_op<op::translate_1to1_match_2_inputs_align_types<opset10::Multiply>>},
        {"aten::narrow", op::translate_narrow},
        {"aten::ne", op::translate_1to1_match_2_inputs_align_types<opset10::NotEqual>},
        {"aten::neg", op::translate_neg},
        {"aten::new_empty", op::translate_new_zeros},
        {"aten::new_full", op::translate_new_full},
        {"aten::new_ones", op::translate_new_ones},
        {"aten::new_zeros", op::translate_new_zeros},
        {"aten::nonzero", op::translate_nonzero},
        {"aten::norm", op::translate_norm},
        {"aten::numel", op::translate_numel},
        {"aten::ones", op::translate_ones},
        {"aten::ones_like", op::translate_ones_like},
        {"aten::pad", op::translate_pad},
        {"aten::permute", op::translate_1to1_match_2_inputs<opset10::Transpose>},
        {"aten::pow", op::translate_pow},
        {"aten::reciprocal", op::translate_reciprocal},
        {"aten::relu", op::translate_1to1_match_1_inputs<opset10::Relu>},
        {"aten::relu_", op::inplace_op<op::translate_1to1_match_1_inputs<opset10::Relu>>},
        {"aten::relu6", op::translate_relu6},
        {"aten::remainder", op::translate_remainder},
        {"aten::repeat", op::translate_repeat},
        {"aten::repeat_interleave", op::translate_repeat_interleave},
        {"aten::reshape", op::translate_reshape},
        {"aten::reshape_as", op::translate_reshape_as},
        {"aten::roll", op::translate_roll},
        {"aten::rsqrt", op::translate_rsqrt},
        {"aten::rsub", op::translate_rsub},
        {"aten::ScalarImplicit", op::skip_node},
        {"aten::scaled_dot_product_attention", op::translate_scaled_dot_product_attention},
        {"aten::select", op::translate_select},
        {"aten::selu", op::translate_selu},
        {"aten::selu_", op::inplace_op<op::translate_selu>},
        {"aten::sigmoid", op::translate_1to1_match_1_inputs_with_fp32_type_alignment<opset10::Sigmoid>},
        {"aten::sigmoid_", op::inplace_op<op::translate_1to1_match_1_inputs<opset10::Sigmoid>>},
        {"aten::sign", op::translate_sign},
        {"aten::silu", op::translate_1to1_match_1_inputs<opset10::Swish>},
        {"aten::silu_", op::inplace_op<op::translate_1to1_match_1_inputs<opset10::Swish>>},
        {"aten::sin", op::translate_1to1_match_1_inputs_with_fp32_type_alignment<opset10::Sin>},
        {"aten::sin_", op::inplace_op<op::translate_1to1_match_1_inputs<opset10::Sin>>},
        {"aten::sinh", op::translate_1to1_match_1_inputs_with_fp32_type_alignment<opset10::Sinh>},
        {"aten::sinh_", op::inplace_op<op::translate_1to1_match_1_inputs<opset10::Sinh>>},
        {"aten::size", op::translate_size},
        {"aten::slice", op::translate_slice},
        {"aten::softmax", op::translate_softmax},
        {"aten::sort", op::translate_sort},
        {"aten::sqrt", op::translate_1to1_match_1_inputs_with_fp32_type_alignment<opset10::Sqrt>},
        {"aten::square", op::translate_square},
        {"aten::squeeze", op::translate_squeeze},
        {"aten::sub", op::translate_sub},
        {"aten::sum", op::translate_sum},
        {"aten::tan", op::translate_1to1_match_1_inputs_with_fp32_type_alignment<opset10::Tan>},
        {"aten::tan_", op::inplace_op<op::translate_1to1_match_1_inputs<opset10::Tan>>},
        {"aten::tanh", op::translate_1to1_match_1_inputs_with_fp32_type_alignment<opset10::Tanh>},
        {"aten::tanh_", op::inplace_op<op::translate_1to1_match_1_inputs<opset10::Tanh>>},
        {"aten::tensor", op::translate_as_tensor},
        {"aten::to", op::translate_to},
        {"aten::topk", op::translate_topk},
        {"aten::transpose", op::translate_transpose},
        {"aten::tril", op::translate_tril},
        {"aten::triu", op::translate_triu},
        {"aten::type_as",
         op::translate_1to1_match_2_inputs<opset10::ConvertLike>},  // TODO: overflow semantics is different
        {"aten::unfold", op::translate_unfold},
        {"aten::unsqueeze", op::translate_1to1_match_2_inputs<opset10::Unsqueeze>},
        {"aten::unsqueeze_", op::inplace_op<op::translate_1to1_match_2_inputs<opset10::Unsqueeze>>},
        {"aten::upsample_bicubic2d", op::translate_upsample_bicubic2d},
        {"aten::upsample_bilinear2d", op::translate_upsample_bilinear2d},
        {"aten::upsample_linear1d", op::translate_upsample_linear1d},
        {"aten::upsample_nearest1d", op::translate_upsample_nearest1d},
        {"aten::upsample_nearest2d", op::translate_upsample_nearest2d},
        {"aten::upsample_nearest3d", op::translate_upsample_nearest3d},
        {"aten::upsample_trilinear3d", op::translate_upsample_trilinear3d},
        {"aten::var", op::translate_var},
        {"aten::var_mean", op::translate_var_mean},
        {"aten::view", op::translate_reshape},
        {"aten::where", op::translate_where},
        {"aten::zeros", op::translate_zeros},
        {"aten::zeros_like", op::translate_zeros_like},
        {"prim::Constant", op::translate_constant},
        {"prim::device", op::translate_constant},
        {"prim::GetAttr", op::translate_get_attr},
        {"prim::If", op::translate_if},
        {"prim::is_cuda", op::return_false_scalar},
        {"prim::ListConstruct", op::translate_list_construct},
        {"prim::Loop", op::translate_loop},
        {"prim::NumToTensor", op::skip_node},  // In openvino we already store number as tensor with shape []
        {"prim::requires_grad", op::return_false_scalar},
        {"prim::PythonOp", op::translate_pythonop},
        {"prim::type", op::skip_node},  // Used with prim::device, pass PtFrameworkNode.
        {"torchvision::deform_conv2d", op::translate_deform_conv},
        {"torchvision::nms", op::translate_nms},
        {"torchvision::roi_align", op::translate_roi_align},
    };
};

const std::map<std::string, CreatorFunction> get_supported_ops_fx() {
    return {
        {"aten._adaptive_avg_pool2d.default", op::translate_1to1_match_2_inputs<opset10::AdaptiveAvgPool>},
        {"aten._softmax.default", op::translate_softmax_fx},
        {"aten._to_copy.default", op::skip_node},
        {"aten._unsafe_view.default", op::translate_reshape},
        {"aten.add.Tensor", op::translate_add},
        {"aten.add_.Tensor", op::translate_add},
        {"aten.addmm.default", op::translate_addmm},
        {"aten.arange.start", op::translate_arange},
        {"aten.avg_pool2d.default", op::translate_avg_poolnd},
        {"aten.bitwise_and.Tensor", op::translate_bitwise_and},
        {"aten.bmm.default", op::translate_1to1_match_2_inputs_align_types<opset10::MatMul>},
        {"aten.cat.default", op::translate_cat_fx},
        {"aten.clone.default", op::skip_node},       // ignore clone operators that are inserted by PyTorch autograd
        {"aten.convolution.default", op::translate_convolution},
        {"aten.copy_.default", op::skip_node},
        {"aten.cos.default", op::translate_1to1_match_1_inputs<opset10::Cos>},
        {"aten.cumsum.default", op::translate_cumsum},
        {"aten.detach.default", op::skip_node},
        {"aten.div.Scalar", op::translate_div},
        {"aten.div.Tensor", op::translate_div},
        {"aten.embedding.default", op::translate_embedding},
        {"aten.empty.memory_format", op::translate_empty},
        {"aten.eq.Tensor", op::translate_1to1_match_2_inputs_align_types<opset10::Equal>},
        {"aten.exp.default", op::translate_1to1_match_1_inputs<opset10::Exp>},
        {"aten.expand.default", op::translate_expand},
        {"aten.full.default", op::translate_full},
        {"aten.gather.default", op::translate_gather},
        {"aten.gelu.default", op::translate_gelu},
        {"aten.gt.Scalar", op::translate_1to1_match_2_inputs_align_types<opset10::Greater>},
        {"aten.hardsigmoid.default", op::translate_1to1_match_1_inputs<opset10::HSigmoid>},
        {"aten.hardswish_.default", op::inplace_op<op::translate_1to1_match_1_inputs<opset10::HSwish>>},
        {"aten.hardtanh_.default", op::inplace_op<op::translate_hardtanh>},
        {"aten.lift_fresh_copy.default", op::skip_node},
        {"aten.log.default", op::translate_log},
        {"aten.logsumexp.default", op::translate_logsumexp},
        {"aten.max.dim", op::translate_max},
        {"aten.max_pool2d_with_indices.default", op::translate_max_poolnd_fx},
        {"aten.mean.dim", op::translate_mean},
        {"aten.mm.default", op::translate_1to1_match_2_inputs<opset10::MatMul>},
        {"aten.mul.Tensor", op::translate_1to1_match_2_inputs_align_types<opset10::Multiply>},
        {"aten.native_batch_norm.default", op::translate_batch_norm_fx},
        {"aten.native_group_norm.default", op::translate_group_norm_fx},
        {"aten.native_layer_norm.default", op::translate_layer_norm},
        {"aten.neg.default", op::translate_neg},
        {"aten.new_ones.default", op::translate_new_ones},
        {"aten.permute.default", op::translate_1to1_match_2_inputs<opset10::Transpose>},
        {"aten.pow.Tensor_Scalar", op::translate_pow},
        {"aten.relu.default", op::translate_1to1_match_1_inputs<opset10::Relu>},
        {"aten.relu_.default", op::inplace_op<op::translate_1to1_match_1_inputs<opset10::Relu>>},
        {"aten.rsub.Scalar", op::translate_rsub},
        {"aten.select.int", op::translate_select},
        {"aten.sigmoid.default", op::translate_1to1_match_1_inputs<opset10::Sigmoid>},
        {"aten.silu.default", op::translate_1to1_match_1_inputs<opset10::Swish>},
        {"aten.silu_.default", op::inplace_op<op::translate_1to1_match_1_inputs<opset10::Swish>>},
        {"aten.sin.default", op::translate_1to1_match_1_inputs<opset10::Sin>},
        {"aten.slice.Tensor", op::translate_slice_fx},
        {"aten.split.Tensor", op::translate_chunk_fx},
        {"aten.stack.default", op::translate_cat_fx},
        {"aten.sub.default", op::translate_sub},
        {"aten.sub.Tensor", op::translate_sub},
        {"aten.t.default", op::translate_transpose_fx},
        {"aten.tanh.default", op::translate_1to1_match_1_inputs<opset10::Tanh>},
        {"aten.transpose.int", op::translate_transpose},
        {"aten.unsqueeze.default", op::translate_1to1_match_2_inputs<opset10::Unsqueeze>},
        {"aten.upsample_nearest2d.default", op::translate_upsample_nearest2d},
        {"aten.view.default", op::translate_reshape},
        {"aten.where.self", op::translate_where},
        {"aten.zeros_like.default", op::translate_zeros_like},
        {"get_attr", op::translate_constant},
        {"prim::Constant", op::translate_constant},
        {"prim::device", op::translate_constant},
        {"prim::GetAttr", op::translate_get_attr},
        {"prim::If", op::translate_if},
        {"prim::is_cuda", op::return_false_scalar},
        {"prim::ListConstruct", op::translate_list_construct},
        {"prim::Loop", op::translate_loop},
        {"prim::NumToTensor", op::skip_node},  // In openvino we already store number as tensor with shape []
        {"prim::PythonOp", op::translate_pythonop},
        {"prim::requires_grad", op::return_false_scalar},
        {"prim::type", op::skip_node},  // Used with prim::device, pass PtFrameworkNode.
        {"torchvision::deform_conv2d", op::translate_deform_conv},
        {"torchvision::nms", op::translate_nms},
        {"torchvision::roi_align", op::translate_roi_align},
        {"<built-in function getitem>", op::translate_getitem}, // TODO: Check if there is any other way to handle this

    };
};


}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
