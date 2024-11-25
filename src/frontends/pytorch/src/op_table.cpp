// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "op_table.hpp"

#include "openvino/opsets/opset10.hpp"
#include "utils.hpp"
#include "utils_quantize.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

#define OP_CONVERTER(op) OutputVector op(const NodeContext& node)

// TorchScript translations
OP_CONVERTER(translate_adaptive_avg_pool3d);
OP_CONVERTER(translate_adaptive_avg_pool2d);
OP_CONVERTER(translate_adaptive_avg_pool1d);
OP_CONVERTER(translate_adaptive_max_pool3d);
OP_CONVERTER(translate_adaptive_max_pool2d);
OP_CONVERTER(translate_adaptive_max_pool1d);
OP_CONVERTER(translate_add);
OP_CONVERTER(translate_add_);
OP_CONVERTER(translate_aminmax);
OP_CONVERTER(translate_mul);
OP_CONVERTER(translate_mul_);
OP_CONVERTER(translate_addcmul);
OP_CONVERTER(translate_addmm);
OP_CONVERTER(translate_alias_copy);
OP_CONVERTER(translate_all);
OP_CONVERTER(translate_any);
OP_CONVERTER(translate_amax);
OP_CONVERTER(translate_amin);
OP_CONVERTER(translate_and);
OP_CONVERTER(translate_arange);
OP_CONVERTER(translate_argmax);
OP_CONVERTER(translate_argsort);
OP_CONVERTER(translate_argmax);
OP_CONVERTER(translate_argmin);
OP_CONVERTER(translate_as_strided);
OP_CONVERTER(translate_as_tensor);
OP_CONVERTER(translate_atan2);
OP_CONVERTER(translate_avg_pool1d);
OP_CONVERTER(translate_avg_pool2d);
OP_CONVERTER(translate_avg_pool3d);
OP_CONVERTER(translate_bool);
OP_CONVERTER(translate_batch_norm);
OP_CONVERTER(translate_bitwise_and);
OP_CONVERTER(translate_bitwise_not);
OP_CONVERTER(translate_bitwise_or);
OP_CONVERTER(translate_bitwise_xor);
OP_CONVERTER(translate_bucketize);
OP_CONVERTER(translate_cat);
OP_CONVERTER(translate_cdist);
OP_CONVERTER(translate_celu);
OP_CONVERTER(translate_channel_shuffle);
OP_CONVERTER(translate_clamp);
OP_CONVERTER(translate_col2im);
OP_CONVERTER(translate_constant);
OP_CONVERTER(translate_conv_transposend);
OP_CONVERTER(translate_conv1d_ext);
OP_CONVERTER(translate_convnd);
OP_CONVERTER(translate_convolution);
OP_CONVERTER(translate_convolution_mode);
OP_CONVERTER(translate_copy_);
OP_CONVERTER(translate_cross);
OP_CONVERTER(translate_cumsum);
OP_CONVERTER(translate_deform_conv);
OP_CONVERTER(translate_derive_index);
OP_CONVERTER(translate_dim);
OP_CONVERTER(translate_div);
OP_CONVERTER(translate_div_);
OP_CONVERTER(translate_dot);
OP_CONVERTER(translate_elu);
OP_CONVERTER(translate_embedding);
OP_CONVERTER(translate_embedding_bag);
OP_CONVERTER(translate_embedding_ext);
OP_CONVERTER(translate_empty);
OP_CONVERTER(translate_empty_like);
OP_CONVERTER(translate_erf);
OP_CONVERTER(translate_erfc);
OP_CONVERTER(translate_expand);
OP_CONVERTER(translate_expand_as);
OP_CONVERTER(translate_expm1);
OP_CONVERTER(translate_eye);
OP_CONVERTER(translate_fake_quantize_per_channel_affine);
OP_CONVERTER(translate_fake_quantize_per_tensor_affine);
OP_CONVERTER(translate_fill);
OP_CONVERTER(translate_fill_diagonal);
OP_CONVERTER(translate_flatten);
OP_CONVERTER(translate_flip);
OP_CONVERTER(translate_floor_divide);
OP_CONVERTER(translate_fmod);
OP_CONVERTER(translate_frobenius_norm);
OP_CONVERTER(translate_full);
OP_CONVERTER(translate_full_like);
OP_CONVERTER(translate_gather);
OP_CONVERTER(translate_gcd);
OP_CONVERTER(translate_gelu);
OP_CONVERTER(translate_get_attr);
OP_CONVERTER(translate_getitem);
OP_CONVERTER(translate_glu);
OP_CONVERTER(translate_grid_sampler);
OP_CONVERTER(translate_group_norm);
OP_CONVERTER(translate_gru);
OP_CONVERTER(translate_hann_window);
OP_CONVERTER(translate_hardtanh);
OP_CONVERTER(translate_if);
OP_CONVERTER(translate_im2col);
OP_CONVERTER(translate_index);
OP_CONVERTER(translate_index_add);
OP_CONVERTER(translate_index_copy_);
OP_CONVERTER(translate_index_put_);
OP_CONVERTER(translate_index_select);
OP_CONVERTER(translate_instance_norm);
OP_CONVERTER(translate_int);
OP_CONVERTER(translate_inverse);
OP_CONVERTER(translate_is_nonzero);
OP_CONVERTER(translate_layer_norm);
OP_CONVERTER(translate_len);
OP_CONVERTER(translate_lerp);
OP_CONVERTER(translate_linalg_cross);
OP_CONVERTER(translate_linalg_norm);
OP_CONVERTER(translate_linalg_matrix_norm);
OP_CONVERTER(translate_linalg_vector_norm);
OP_CONVERTER(translate_linear);
OP_CONVERTER(translate_linspace);
OP_CONVERTER(translate_list_construct);
OP_CONVERTER(translate_list_unpack);
OP_CONVERTER(translate_log1p);
OP_CONVERTER(translate_log_sigmoid);
OP_CONVERTER(translate_log_softmax);
OP_CONVERTER(translate_log2);
OP_CONVERTER(translate_log10);
OP_CONVERTER(translate_logsumexp);
OP_CONVERTER(translate_loop);
OP_CONVERTER(translate_lstm);
OP_CONVERTER(translate_masked_fill);
OP_CONVERTER(translate_masked_scatter);
OP_CONVERTER(translate_masked_select);
OP_CONVERTER(translate_max);
OP_CONVERTER(translate_maximum);
OP_CONVERTER(translate_max_pool1d);
OP_CONVERTER(translate_max_pool2d);
OP_CONVERTER(translate_max_pool3d);
OP_CONVERTER(translate_mean);
OP_CONVERTER(translate_meshgrid);
OP_CONVERTER(translate_min);
OP_CONVERTER(translate_minimum);
OP_CONVERTER(translate_movedim);
OP_CONVERTER(translate_multinomial);
OP_CONVERTER(translate_narrow);
OP_CONVERTER(translate_native_multi_head_attention);
OP_CONVERTER(translate_neg);
OP_CONVERTER(translate_new_full);
OP_CONVERTER(translate_new_ones);
OP_CONVERTER(translate_new_zeros);
OP_CONVERTER(translate_nms);
OP_CONVERTER(translate_nonzero);
OP_CONVERTER(translate_norm);
OP_CONVERTER(translate_normal);
OP_CONVERTER(translate_normal_);
OP_CONVERTER(translate_not);
OP_CONVERTER(translate_numel);
OP_CONVERTER(translate_one_hot);
OP_CONVERTER(translate_ones);
OP_CONVERTER(translate_ones_like);
OP_CONVERTER(translate_or);
OP_CONVERTER(translate_bitwise_xor);
OP_CONVERTER(translate_outer);
OP_CONVERTER(translate_pack_padded_sequence);
OP_CONVERTER(translate_pad);
OP_CONVERTER(translate_pad_packed_sequence);
OP_CONVERTER(translate_pairwise_distance);
OP_CONVERTER(translate_pixel_shuffle);
OP_CONVERTER(translate_pixel_unshuffle);
OP_CONVERTER(translate_pow);
OP_CONVERTER(translate_prod);
OP_CONVERTER(translate_pythonop);
OP_CONVERTER(translate_quantize_per_channel);
OP_CONVERTER(translate_quantize_per_tensor);
OP_CONVERTER(translate_quantized_add);
OP_CONVERTER(translate_quantized_add_relu);
OP_CONVERTER(translate_quantized_hardswish);
OP_CONVERTER(translate_quantized_mul);
OP_CONVERTER(translate_range_length);
OP_CONVERTER(translate_rand);
OP_CONVERTER(translate_randn);
OP_CONVERTER(translate_randint);
OP_CONVERTER(translate_rand_like);
OP_CONVERTER(translate_randn_like);
OP_CONVERTER(translate_reciprocal);
OP_CONVERTER(translate_relu6);
OP_CONVERTER(translate_remainder);
OP_CONVERTER(translate_repeat_interleave);
OP_CONVERTER(translate_reshape);
OP_CONVERTER(translate_reshape_as);
OP_CONVERTER(translate_rnn);
OP_CONVERTER(translate_roi_align);
OP_CONVERTER(translate_roll);
OP_CONVERTER(translate_round);
OP_CONVERTER(translate_rsqrt);
OP_CONVERTER(translate_rsub);
OP_CONVERTER(translate_scaled_dot_product_attention);
OP_CONVERTER(translate_scatter);
OP_CONVERTER(translate_scatter_add);
OP_CONVERTER(translate_scatter_reduce);
OP_CONVERTER(translate_select);
OP_CONVERTER(translate_set_item);
OP_CONVERTER(translate_selu);
OP_CONVERTER(translate_shape_as_tensor);
OP_CONVERTER(translate_sign);
OP_CONVERTER(translate_size);
OP_CONVERTER(translate_slice);
OP_CONVERTER(translate_softmax);
OP_CONVERTER(translate_sort);
OP_CONVERTER(translate_square);
OP_CONVERTER(translate_squeeze);
OP_CONVERTER(translate_std);
OP_CONVERTER(translate_std_mean);
OP_CONVERTER(translate_stft);
OP_CONVERTER(translate_sub);
OP_CONVERTER(translate_sub_);
OP_CONVERTER(translate_sum);
OP_CONVERTER(translate_t);
OP_CONVERTER(translate_take_along_dim);
OP_CONVERTER(translate_to);
OP_CONVERTER(translate_topk);
OP_CONVERTER(translate_transpose);
OP_CONVERTER(translate_tril);
OP_CONVERTER(translate_triu);
OP_CONVERTER(translate_tuple_index);
OP_CONVERTER(translate_unflatten);
OP_CONVERTER(translate_unfold);
OP_CONVERTER(translate_upsample_bicubic2d);
OP_CONVERTER(translate_upsample_bilinear2d);
OP_CONVERTER(translate_upsample_bicubic2d_aa);
OP_CONVERTER(translate_upsample_bilinear2d_aa);
OP_CONVERTER(translate_upsample_linear1d);
OP_CONVERTER(translate_upsample_nearest1d);
OP_CONVERTER(translate_upsample_nearest2d);
OP_CONVERTER(translate_upsample_nearest3d);
OP_CONVERTER(translate_upsample_trilinear3d);
OP_CONVERTER(translate_var);
OP_CONVERTER(translate_var_mean);
OP_CONVERTER(translate_weight_norm);
OP_CONVERTER(translate_where);
OP_CONVERTER(translate_zeros);
OP_CONVERTER(translate_zeros_like);
OP_CONVERTER(translate_quantized_cat);
OP_CONVERTER(translate_quantized_convnd);
OP_CONVERTER(translate_quantized_convnd_relu);
OP_CONVERTER(translate_quantized_linear);
OP_CONVERTER(translate_xor);
// Torch FX Translations
OP_CONVERTER(translate_adaptive_max_pool1d_fx);
OP_CONVERTER(translate_adaptive_max_pool2d_fx);
OP_CONVERTER(translate_adaptive_max_pool3d_fx);
OP_CONVERTER(translate_addcmul_fx);
OP_CONVERTER(translate_addmm_fx);
OP_CONVERTER(translate_any_fx);
OP_CONVERTER(translate_arange_fx);
OP_CONVERTER(translate_batch_norm_legit_fx);
OP_CONVERTER(translate_batch_norm_legit_no_training_fx);
OP_CONVERTER(translate_batch_norm_legit_no_stats_fx);
OP_CONVERTER(translate_cat_fx);
OP_CONVERTER(translate_constant_pad_nd_fx);
OP_CONVERTER(translate_copy_fx);
OP_CONVERTER(translate_cumsum_fx);
OP_CONVERTER(translate_chunk_fx);
OP_CONVERTER(translate_div_fx);
OP_CONVERTER(translate_div_fx_);
OP_CONVERTER(translate_embedding_bag_fx);
OP_CONVERTER(translate_expand_fx);
OP_CONVERTER(translate_eye_fx);
OP_CONVERTER(translate_fake_quantize_per_channel_affine_fx);
OP_CONVERTER(translate_fake_quantize_per_tensor_affine_fx);
OP_CONVERTER(translate_full_fx);
OP_CONVERTER(translate_full_like_fx);
OP_CONVERTER(translate_gelu_fx);
OP_CONVERTER(translate_group_norm_fx);
OP_CONVERTER(translate_index_fx);
OP_CONVERTER(translate_layer_norm_fx);
OP_CONVERTER(translate_leaky_relu_fx);
OP_CONVERTER(translate_log_sigmoid_fx);
OP_CONVERTER(translate_log_softmax_fx);
OP_CONVERTER(translate_max_dim_fx);
OP_CONVERTER(translate_max_pool2d_fx);
OP_CONVERTER(translate_max_pool3d_fx);
OP_CONVERTER(translate_mean_fx);
OP_CONVERTER(translate_min_dim_fx);
OP_CONVERTER(translate_new_full_fx);
OP_CONVERTER(translate_new_ones_fx);
OP_CONVERTER(translate_new_zeros_fx);
OP_CONVERTER(translate_ones_fx);
OP_CONVERTER(translate_ones_like_fx);
OP_CONVERTER(translate_reflection_pad_nd_fx);
OP_CONVERTER(translate_reshape_fx);
OP_CONVERTER(translate_rsub_fx);
OP_CONVERTER(translate_scalar_tensor_fx);
OP_CONVERTER(translate_scaled_dot_product_attention_fx);
OP_CONVERTER(translate_search_sorted);
OP_CONVERTER(translate_select_scatter_fx);
OP_CONVERTER(translate_slice_fx);
OP_CONVERTER(translate_slice_scatter_fx);
OP_CONVERTER(translate_softmax_fx);
OP_CONVERTER(translate_sort_fx);
OP_CONVERTER(translate_split_with_sizes_fx);
OP_CONVERTER(translate_stack_fx);
OP_CONVERTER(translate_sub_fx);
OP_CONVERTER(translate_sum_fx);
OP_CONVERTER(translate_std_fx);
OP_CONVERTER(translate_topk_fx);
OP_CONVERTER(translate_to_fx);
OP_CONVERTER(translate_transpose_fx);
OP_CONVERTER(translate_quantize_per_channel_fx);
OP_CONVERTER(translate_quantize_per_tensor_fx);
OP_CONVERTER(translate_var_fx);
OP_CONVERTER(translate_var_mean_fx);
OP_CONVERTER(translate_unbind_int_fx);
OP_CONVERTER(translate_unique2);
OP_CONVERTER(translate_zeros_fx);
OP_CONVERTER(translate_zeros_like_fx);

}  // namespace op

// Supported ops for TorchScript
const std::unordered_map<std::string, CreatorFunction> get_supported_ops_ts() {
    return {
        {"aten::__and__", op::translate_bitwise_and},
        {"aten::__iand__", op::inplace_op<op::translate_bitwise_and>},
        {"aten::__derive_index", op::translate_derive_index},
        {"aten::__getitem__", op::translate_getitem},
        {"aten::__not__", op::translate_1to1_match_1_inputs<opset10::LogicalNot>},
        {"aten::__or__", op::translate_bitwise_or},
        {"aten::__ior__", op::inplace_op<op::translate_bitwise_or>},
        {"aten::__range_length", op::translate_range_length},
        {"aten::__xor__", op::translate_bitwise_xor},
        {"aten::__ixor__", op::inplace_op<op::translate_bitwise_xor>},
        {"aten::_convolution", op::translate_convolution},
        {"aten::_convolution_mode", op::translate_convolution_mode},
        {"aten::_native_multi_head_attention", op::translate_native_multi_head_attention},
        {"aten::_pack_padded_sequence", op::translate_pack_padded_sequence},
        {"aten::_pad_packed_sequence", op::translate_pad_packed_sequence},
        {"aten::_set_item", op::translate_set_item},
        {"aten::_shape_as_tensor", op::translate_shape_as_tensor},
        {"aten::_unique2", op::translate_unique2},
        {"aten::_upsample_bicubic2d_aa", op::translate_upsample_bicubic2d_aa},
        {"aten::_upsample_bilinear2d_aa", op::translate_upsample_bilinear2d_aa},
        {"aten::_weight_norm", op::translate_weight_norm},
        {"aten::abs", op::optional_out<op::translate_1to1_match_1_inputs<opset10::Abs>, 1>},
        {"aten::abs_", op::inplace_op<op::translate_1to1_match_1_inputs<opset10::Abs>>},
        {"aten::acos", op::optional_out<op::translate_1to1_match_1_inputs_with_fp32_type_alignment<opset10::Acos>, 1>},
        {"aten::acos_", op::inplace_op<op::translate_1to1_match_1_inputs<opset10::Acos>>},
        {"aten::acosh",
         op::optional_out<op::translate_1to1_match_1_inputs_with_fp32_type_alignment<opset10::Acosh>, 1>},
        {"aten::acosh_", op::inplace_op<op::translate_1to1_match_1_inputs<opset10::Acosh>>},
        {"aten::adaptive_avg_pool1d", op::quantizable_op<op::translate_adaptive_avg_pool1d>},
        {"aten::adaptive_avg_pool2d", op::quantizable_op<op::translate_adaptive_avg_pool2d>},
        {"aten::adaptive_avg_pool3d", op::quantizable_op<op::translate_adaptive_avg_pool3d>},
        {"aten::adaptive_max_pool1d", op::quantizable_op<op::translate_adaptive_max_pool1d>},
        {"aten::adaptive_max_pool2d", op::quantizable_op<op::translate_adaptive_max_pool2d>},
        {"aten::adaptive_max_pool3d", op::quantizable_op<op::translate_adaptive_max_pool3d>},
        {"aten::add", op::translate_add},
        {"aten::add_", op::translate_add_},
        {"aten::addcmul", op::translate_addcmul},
        {"aten::addmm", op::translate_addmm},
        {"aten::alias", op::skip_node},
        {"aten::alias_copy", op::translate_alias_copy},
        {"aten::all", op::translate_all},
        {"aten::amax", op::translate_amax},
        {"aten::amin", op::translate_amin},
        {"aten::aminmax", op::translate_aminmax},
        {"aten::any", op::translate_any},
        // aten::append - Supported in limited set of patterns
        {"aten::arange", op::translate_arange},
        {"aten::argmax", op::translate_argmax},
        {"aten::argmin", op::translate_argmin},
        {"aten::argsort", op::translate_argsort},
        {"aten::as_strided", op::translate_as_strided},
        {"aten::as_tensor", op::translate_as_tensor},
        {"aten::asin", op::optional_out<op::translate_1to1_match_1_inputs_with_fp32_type_alignment<opset10::Asin>, 1>},
        {"aten::asin_", op::inplace_op<op::translate_1to1_match_1_inputs<opset10::Asin>>},
        {"aten::asinh",
         op::optional_out<op::translate_1to1_match_1_inputs_with_fp32_type_alignment<opset10::Asinh>, 1>},
        {"aten::asinh_", op::inplace_op<op::translate_1to1_match_1_inputs<opset10::Asinh>>},
        {"aten::atan", op::optional_out<op::translate_1to1_match_1_inputs_with_fp32_type_alignment<opset10::Atan>, 1>},
        {"aten::atan_", op::inplace_op<op::translate_1to1_match_1_inputs<opset10::Atan>>},
        {"aten::atanh",
         op::optional_out<op::translate_1to1_match_1_inputs_with_fp32_type_alignment<opset10::Atanh>, 1>},
        {"aten::atanh_", op::inplace_op<op::translate_1to1_match_1_inputs<opset10::Atanh>>},
        {"aten::atan2", op::translate_atan2},
        {"aten::avg_pool1d", op::quantizable_op<op::translate_avg_pool1d>},
        {"aten::avg_pool2d", op::quantizable_op<op::translate_avg_pool2d>},
        {"aten::avg_pool3d", op::quantizable_op<op::translate_avg_pool3d>},
        {"aten::baddbmm", op::translate_addmm},
        {"aten::batch_norm", op::translate_batch_norm},
        {"aten::bitwise_and", op::translate_bitwise_and},
        {"aten::bitwise_not", op::translate_bitwise_not},
        {"aten::bitwise_or", op::translate_bitwise_or},
        {"aten::bitwise_xor", op::translate_bitwise_xor},
        {"aten::bmm", op::translate_1to1_match_2_inputs<opset10::MatMul>},
        {"aten::Bool", op::translate_bool},
        // aten::broadcast_tensors - Supported in limited set of patterns
        {"aten::broadcast_to", op::translate_expand},
        {"aten::bucketize", op::translate_bucketize},
        {"aten::cat", op::translate_cat},
        {"aten::cdist", op::translate_cdist},
        {"aten::ceil", op::optional_out<op::translate_1to1_match_1_inputs<opset10::Ceiling>, 1>},
        {"aten::ceil_", op::inplace_op<op::translate_1to1_match_1_inputs<opset10::Ceiling>>},
        {"aten::celu", op::translate_celu},
        {"aten::channel_shuffle", op::translate_channel_shuffle},
        // aten::chunk - Supported in limited set of patterns
        {"aten::clamp", op::translate_clamp},
        {"aten::clamp_max", op::translate_1to1_match_2_inputs_align_types<opset10::Minimum>},
        {"aten::clamp_min", op::translate_1to1_match_2_inputs_align_types<opset10::Maximum>},
        {"aten::clip", op::translate_clamp},
        {"aten::clone", op::skip_node},  // ignore clone operators that are inserted by PyTorch autograd
        {"aten::col2im", op::translate_col2im},
        // aten::complex - Supported in limited set of patterns
        {"aten::concat", op::translate_cat},
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
        {"aten::copy_", op::translate_copy_},
        {"aten::cos", op::optional_out<op::translate_1to1_match_1_inputs_with_fp32_type_alignment<opset10::Cos>, 1>},
        {"aten::cos_", op::inplace_op<op::translate_1to1_match_1_inputs<opset10::Cos>>},
        {"aten::cosh", op::optional_out<op::translate_1to1_match_1_inputs_with_fp32_type_alignment<opset10::Cosh>, 1>},
        {"aten::cosh_", op::inplace_op<op::translate_1to1_match_1_inputs<opset10::Cosh>>},
        {"aten::cross", op::translate_cross},
        {"aten::cumsum", op::translate_cumsum},
        {"aten::detach", op::skip_node},
        {"aten::dequantize", op::skip_node},  // we convert model to fp32 using FQ, so dequantization is not needed
        {"aten::dim", op::translate_dim},
        {"aten::div", op::translate_div},
        {"aten::div_", op::translate_div_},
        {"aten::dot", op::translate_dot},
        {"aten::dropout", op::skip_node},
        {"aten::dropout_", op::skip_node},
        // aten::einsum - Supported in limited set of patterns
        {"aten::elu", op::translate_elu},
        {"aten::embedding", op::translate_embedding},
        {"aten::embedding_bag", op::translate_embedding_bag},
        {"aten::empty", op::translate_empty},
        {"aten::empty_like", op::translate_empty_like},
        {"aten::eq", op::translate_1to1_match_2_inputs_align_types<opset10::Equal>},
        {"aten::erf", op::translate_erf},
        {"aten::erfc", op::translate_erfc},
        {"aten::exp", op::optional_out<op::translate_1to1_match_1_inputs_with_fp32_type_alignment<opset10::Exp>, 1>},
        {"aten::exp_", op::inplace_op<op::translate_1to1_match_1_inputs_with_fp32_type_alignment<opset10::Exp>>},
        {"aten::expand", op::translate_expand},
        {"aten::expand_as", op::translate_expand_as},
        {"aten::expm1", op::translate_expm1},
        {"aten::eye", op::translate_eye},
        {"aten::fake_quantize_per_channel_affine", op::translate_fake_quantize_per_channel_affine},
        {"aten::fake_quantize_per_tensor_affine", op::translate_fake_quantize_per_tensor_affine},
        {"aten::feature_dropout", op::skip_node},
        // aten::fft_irfftn - Supported in limited set of patterns
        // aten::fft_rfftn - Supported in limited set of patterns
        {"aten::fill", op::translate_fill},
        {"aten::fill_diagonal", op::translate_fill_diagonal},
        {"aten::flatten", op::quantizable_op<op::translate_flatten>},
        {"aten::flip", op::translate_flip},
        {"aten::floor", op::optional_out<op::translate_1to1_match_1_inputs<opset10::Floor>, 1>},
        {"aten::floor_", op::inplace_op<op::translate_1to1_match_1_inputs<opset10::Floor>>},
        {"aten::floor_divide", op::translate_floor_divide},
        {"aten::floordiv", op::translate_floor_divide},
        {"aten::fmod", op::translate_fmod},
        {"aten::frobenius_norm", op::translate_frobenius_norm},
        {"aten::full", op::translate_full},
        {"aten::full_like", op::translate_full_like},
        {"aten::gather", op::translate_gather},
        {"aten::gcd", op::translate_gcd},
        {"aten::ge", op::translate_1to1_match_2_inputs_align_types<opset10::GreaterEqual>},
        {"aten::gelu", op::translate_gelu},
        {"aten::glu", op::translate_glu},
        {"aten::grid_sampler", op::translate_grid_sampler},
        {"aten::group_norm", op::translate_group_norm},
        {"aten::gru", op::translate_gru},
        {"aten::gt", op::translate_1to1_match_2_inputs_align_types<opset10::Greater>},
        {"aten::hann_window", op::translate_hann_window},
        {"aten::hardsigmoid", op::quantizable_op<op::translate_1to1_match_1_inputs<opset10::HSigmoid>>},
        {"aten::hardswish", op::quantizable_op<op::translate_1to1_match_1_inputs<opset10::HSwish>>},
        {"aten::hardtanh", op::quantizable_op<op::translate_hardtanh>},
        {"aten::im2col", op::translate_im2col},
        // aten::imag - Supported in limited set of patterns
        // aten::index - Supported in limited set of patterns
        {"aten::index_copy_", op::inplace_op<op::translate_index_copy_>},
        {"aten::index_put_", op::inplace_op<op::translate_index_put_>},
        {"aten::index_add", op::translate_index_add},
        {"aten::index_select", op::translate_index_select},
        {"aten::instance_norm", op::translate_instance_norm},
        {"aten::inverse", op::translate_inverse},
        {"aten::Int", op::translate_int},
        {"aten::IntImplicit", op::translate_int},
        {"aten::is_grad_enabled", op::return_false_scalar},
        {"aten::is_nonzero", op::translate_is_nonzero},
        {"aten::isfinite", op::translate_1to1_match_1_inputs<opset10::IsFinite>},
        {"aten::isinf", op::translate_1to1_match_1_inputs<opset10::IsInf>},
        {"aten::isnan", op::translate_1to1_match_1_inputs<opset10::IsNaN>},
        {"aten::item", op::translate_1to1_match_1_inputs<opset10::Squeeze>},
        {"aten::layer_norm", op::translate_layer_norm},
        {"aten::le", op::translate_1to1_match_2_inputs_align_types<opset10::LessEqual>},
        {"aten::leaky_relu", op::translate_1to1_match_2_inputs<opset10::PRelu>},
        {"aten::len", op::translate_len},
        {"aten::lerp", op::translate_lerp},
        // lift op is torchscript specific op responsible for tensors coping with guarantee of new memory allocation
        {"aten::lift", op::skip_node},
        {"aten::lift_fresh", op::skip_node},
        {"aten::lift_fresh_copy", op::skip_node},
        {"aten::linalg_cross", op::translate_linalg_cross},
        {"aten::linalg_inv", op::translate_inverse},
        {"aten::linalg_norm", op::translate_linalg_norm},
        {"aten::linalg_matrix_norm", op::translate_linalg_matrix_norm},
        {"aten::linalg_vector_norm", op::translate_linalg_vector_norm},
        {"aten::linear", op::translate_linear},
        {"aten::linspace", op::translate_linspace},
        {"aten::log", op::optional_out<op::translate_1to1_match_1_inputs_with_fp32_type_alignment<opset10::Log>, 1>},
        {"aten::log_", op::inplace_op<op::translate_1to1_match_1_inputs<opset10::Log>>},
        {"aten::logical_and", op::translate_and},
        {"aten::logical_or", op::translate_or},
        {"aten::logical_not", op::translate_not},
        {"aten::logical_xor", op::translate_xor},
        {"aten::log_sigmoid", op::translate_log_sigmoid},
        {"aten::log_softmax", op::translate_log_softmax},
        {"aten::log1p", op::optional_out<op::translate_log1p, 1>},
        {"aten::log1p_", op::inplace_op<op::translate_log1p>},
        {"aten::log2", op::optional_out<op::translate_log2, 1>},
        {"aten::log2_", op::inplace_op<op::translate_log2>},
        {"aten::log10", op::optional_out<op::translate_log10, 1>},
        {"aten::log10_", op::inplace_op<op::translate_log10>},
        {"aten::lstm", op::translate_lstm},
        {"aten::lt", op::translate_1to1_match_2_inputs_align_types<opset10::Less>},
        {"aten::masked_fill", op::translate_masked_fill},
        {"aten::masked_scatter", op::translate_masked_scatter},
        {"aten::masked_select", op::translate_masked_select},
        {"aten::matmul", op::translate_1to1_match_2_inputs<opset10::MatMul>},
        {"aten::max", op::translate_max},
        {"aten::mv", op::translate_1to1_match_2_inputs<opset10::MatMul>},
        {"aten::maximum", op::translate_maximum},
        {"aten::max_pool1d", op::quantizable_op<op::translate_max_pool1d>},
        {"aten::max_pool1d_with_indices", op::quantizable_op<op::translate_max_pool1d>},
        {"aten::max_pool2d", op::quantizable_op<op::translate_max_pool2d>},
        {"aten::max_pool2d_with_indices", op::quantizable_op<op::translate_max_pool2d>},
        {"aten::max_pool3d", op::quantizable_op<op::translate_max_pool3d>},
        {"aten::max_pool3d_with_indices", op::quantizable_op<op::translate_max_pool3d>},
        {"aten::mean", op::quantizable_op<op::translate_mean>},
        {"aten::meshgrid", op::translate_meshgrid},
        {"aten::min", op::translate_min},
        {"aten::minimum", op::translate_minimum},
        {"aten::mish", op::translate_1to1_match_1_inputs_with_fp32_type_alignment<opset10::Mish>},
        {"aten::mm", op::translate_1to1_match_2_inputs<opset10::MatMul>},
        {"aten::movedim", op::translate_movedim},
        {"aten::mul", op::translate_mul},
        {"aten::mul_", op::translate_mul_},
        {"aten::multiply", op::translate_mul},
        {"aten::multiply_", op::translate_mul_},
        {"aten::multinomial", op::translate_multinomial},
        {"aten::narrow", op::translate_narrow},
        {"aten::ne", op::translate_1to1_match_2_inputs_align_types<opset10::NotEqual>},
        {"aten::neg", op::translate_neg},
        {"aten::new_empty", op::translate_new_zeros},
        {"aten::new_full", op::translate_new_full},
        {"aten::new_ones", op::translate_new_ones},
        {"aten::new_zeros", op::translate_new_zeros},
        {"aten::nonzero", op::translate_nonzero},
        // aten::nonzero_numpy - Supported in limited set of patterns
        {"aten::norm", op::translate_norm},
        {"aten::normal", op::translate_normal},
        {"aten::normal_", op::translate_normal_},
        {"aten::numel", op::translate_numel},
        {"aten::numpy_T", op::translate_t},
        {"aten::one_hot", op::translate_one_hot},
        {"aten::ones", op::translate_ones},
        {"aten::ones_like", op::translate_ones_like},
        {"aten::outer", op::translate_outer},
        {"aten::pad", op::translate_pad},
        {"aten::pairwise_distance", op::translate_pairwise_distance},
        {"aten::permute", op::translate_1to1_match_2_inputs<opset10::Transpose>},
        {"aten::pixel_shuffle", op::translate_pixel_shuffle},
        {"aten::pixel_unshuffle", op::translate_pixel_unshuffle},
        {"aten::prelu", op::translate_1to1_match_2_inputs<opset10::PRelu>},
        {"aten::pow", op::translate_pow},
        {"aten::pow_", op::translate_pow},
        {"aten::prod", op::translate_prod},
        {"aten::quantize_per_channel", op::translate_quantize_per_channel},
        {"aten::quantize_per_tensor", op::translate_quantize_per_tensor},
        {"aten::rand", op::translate_rand},
        {"aten::rand_like", op::translate_rand_like},
        {"aten::randint", op::translate_randint},
        {"aten::randn", op::translate_randn},
        {"aten::randn_like", op::translate_randn_like},
        // aten::real - Supported in limited set of patterns
        {"aten::reciprocal", op::optional_out<op::translate_reciprocal, 1>},
        {"aten::reciprocal_", op::inplace_op<op::translate_reciprocal>},
        // aten::reflection_pad2d - Supported in limited set of patterns
        {"aten::relu", op::optional_out<op::translate_1to1_match_1_inputs<opset10::Relu>, 1>},
        {"aten::relu_", op::inplace_op<op::translate_1to1_match_1_inputs<opset10::Relu>>},
        {"aten::relu6", op::translate_relu6},
        {"aten::remainder", op::translate_remainder},
        {"aten::repeat", op::translate_1to1_match_2_inputs<opset10::Tile>},
        {"aten::repeat_interleave", op::translate_repeat_interleave},
        {"aten::reshape", op::translate_reshape},
        {"aten::reshape_as", op::translate_reshape_as},
        // TO DO: enable behaviour for resolve_conj and resolve_neg complex tensors,
        // when complex dtype will be supported
        // for real dtypes, these operations return input tensor without changes and can be skipped
        {"aten::resolve_conj", op::skip_node},
        {"aten::resolve_neg", op::skip_node},
        {"aten::rnn_relu", op::translate_rnn},
        {"aten::rnn_tanh", op::translate_rnn},
        {"aten::roll", op::translate_roll},
        {"aten::round", op::translate_round},
        {"aten::rsqrt", op::optional_out<op::translate_rsqrt, 1>},
        {"aten::rsqrt_", op::inplace_op<op::translate_rsqrt>},
        {"aten::rsub", op::translate_rsub},
        {"aten::searchsorted", op::translate_search_sorted},
        {"aten::ScalarImplicit", op::skip_node},
        {"aten::scaled_dot_product_attention", op::translate_scaled_dot_product_attention},
        {"aten::scatter", op::translate_scatter},
        {"aten::scatter_add", op::translate_scatter_add},
        {"aten::scatter_reduce", op::translate_scatter_reduce},
        {"aten::select", op::quantizable_op<op::translate_select>},
        {"aten::selu", op::translate_selu},
        {"aten::sigmoid",
         op::optional_out<op::translate_1to1_match_1_inputs_with_fp32_type_alignment<opset10::Sigmoid>, 1>},
        {"aten::sigmoid_", op::inplace_op<op::translate_1to1_match_1_inputs<opset10::Sigmoid>>},
        {"aten::sign", op::translate_sign},
        {"aten::silu", op::translate_1to1_match_1_inputs<opset10::Swish>},
        {"aten::sin", op::optional_out<op::translate_1to1_match_1_inputs_with_fp32_type_alignment<opset10::Sin>, 1>},
        {"aten::sin_", op::inplace_op<op::translate_1to1_match_1_inputs<opset10::Sin>>},
        {"aten::sinh", op::optional_out<op::translate_1to1_match_1_inputs_with_fp32_type_alignment<opset10::Sinh>, 1>},
        {"aten::sinh_", op::inplace_op<op::translate_1to1_match_1_inputs<opset10::Sinh>>},
        {"aten::size", op::translate_size},
        {"aten::slice", op::quantizable_op<op::translate_slice>},
        {"aten::softmax", op::translate_softmax},
        {"aten::softplus", op::translate_1to1_match_1_inputs<opset10::SoftPlus>},
        {"aten::sort", op::translate_sort},
        // aten::split - Supported in limited set of patterns
        // aten::split_with_sizes - Supported in limited set of patterns
        {"aten::sqrt", op::optional_out<op::translate_1to1_match_1_inputs_with_fp32_type_alignment<opset10::Sqrt>, 1>},
        {"aten::sqrt_", op::inplace_op<op::translate_1to1_match_1_inputs_with_fp32_type_alignment<opset10::Sqrt>>},
        {"aten::square", op::translate_square},
        {"aten::squeeze", op::quantizable_op<op::translate_squeeze>},
        // aten::stack - Supported in limited set of patterns
        {"aten::std", op::translate_std},
        {"aten::std_mean", op::translate_std_mean},
        {"aten::stft", op::translate_stft},
        {"aten::sub", op::translate_sub},
        {"aten::sub_", op::translate_sub_},
        {"aten::sum", op::translate_sum},
        {"aten::swapaxes", op::quantizable_op<op::translate_transpose>},
        {"aten::t", op::translate_t},
        {"aten::take_along_dim", op::translate_take_along_dim},
        {"aten::tan", op::optional_out<op::translate_1to1_match_1_inputs_with_fp32_type_alignment<opset10::Tan>, 1>},
        {"aten::tan_", op::inplace_op<op::translate_1to1_match_1_inputs<opset10::Tan>>},
        {"aten::tanh", op::optional_out<op::translate_1to1_match_1_inputs_with_fp32_type_alignment<opset10::Tanh>, 1>},
        {"aten::tanh_", op::inplace_op<op::translate_1to1_match_1_inputs<opset10::Tanh>>},
        {"aten::tensor", op::translate_as_tensor},
        // aten::tensor_split - Supported in limited set of patterns
        {"aten::tile", op::translate_1to1_match_2_inputs<opset10::Tile>},
        {"aten::to", op::translate_to},
        {"aten::topk", op::translate_topk},
        {"aten::transpose", op::quantizable_op<op::translate_transpose>},
        {"aten::tril", op::translate_tril},
        {"aten::triu", op::translate_triu},
        {"aten::type_as",
         op::translate_1to1_match_2_inputs<opset10::ConvertLike>},  // TODO: overflow semantics is different
        // aten::unbind - Supported in limited set of patterns
        {"aten::unflatten", op::translate_unflatten},
        {"aten::unfold", op::translate_unfold},
        // aten::unsafe_chunk - Supported in limited set of patterns
        {"aten::unsqueeze", op::quantizable_op<op::translate_1to1_match_2_inputs<opset10::Unsqueeze>>},
        {"aten::upsample_bicubic2d", op::translate_upsample_bicubic2d},
        {"aten::upsample_bilinear2d", op::translate_upsample_bilinear2d},
        {"aten::upsample_linear1d", op::translate_upsample_linear1d},
        {"aten::upsample_nearest1d", op::translate_upsample_nearest1d},
        {"aten::upsample_nearest2d", op::translate_upsample_nearest2d},
        {"aten::upsample_nearest3d", op::translate_upsample_nearest3d},
        {"aten::upsample_trilinear3d", op::translate_upsample_trilinear3d},
        {"aten::var", op::translate_var},
        {"aten::var_mean", op::translate_var_mean},
        {"aten::view", op::quantizable_op<op::translate_reshape>},
        {"aten::view_as", op::translate_reshape_as},
        {"aten::wait", op::skip_node},
        {"aten::where", op::translate_where},
        {"aten::zero", op::translate_zeros_like},
        {"aten::zeros", op::translate_zeros},
        {"aten::zeros_like", op::translate_zeros_like},
        {"ov_ext::embedding", op::translate_embedding_ext},
        {"ov_ext::conv1d", op::translate_conv1d_ext},
        {"ov_ext::linear", op::translate_linear},
        {"prim::Constant", op::translate_constant},
        {"prim::device", op::translate_constant},
        // prim::DictConstruct - Supported in limited set of patterns
        {"prim::fork", op::translate_pythonop},
        {"prim::GetAttr", op::translate_get_attr},
        {"prim::If", op::translate_if},
        {"prim::is_cuda", op::return_false_scalar},
        {"prim::ListConstruct", op::translate_list_construct},
        {"prim::ListUnpack", op::translate_list_unpack},
        {"prim::Loop", op::translate_loop},
        // prim::max - Supported in limited set of patterns
        // prim::min - Supported in limited set of patterns
        {"prim::NumToTensor", op::skip_node},  // In openvino we already store number as tensor with shape []
        {"prim::PythonOp", op::translate_pythonop},
        {"prim::requires_grad", op::return_false_scalar},
        // prim::TupleConstruct - Supported in limited set of patterns
        {"prim::TupleIndex", op::translate_tuple_index},
        // prim::TupleUnpack - Supported in limited set of patterns
        {"prim::type", op::skip_node},  // Used with prim::device, pass PtFrameworkNode.
        {"quantized::add", op::translate_quantized_add},
        {"quantized::add_relu", op::translate_quantized_add_relu},
        {"quantized::cat", op::translate_quantized_cat},
        {"quantized::conv2d", op::translate_quantized_convnd},
        {"quantized::conv2d_relu", op::translate_quantized_convnd_relu},
        {"quantized::hardswish", op::translate_quantized_hardswish},
        {"quantized::linear", op::translate_quantized_linear},
        {"quantized::mul", op::translate_quantized_mul},
        {"torchvision::deform_conv2d", op::translate_deform_conv},
        {"torchvision::nms", op::translate_nms},
        {"torchvision::roi_align", op::translate_roi_align},
    };
};

const std::unordered_map<std::string, CreatorFunction> get_supported_ops_fx() {
    return {
        {"<built-in function add>", op::translate_add},
        {"<built-in function floordiv>", op::translate_floor_divide},
        {"<built-in function getitem>", op::translate_getitem},  // TODO: Check if there is any other way to handle this
        {"<built-in function mul>", op::translate_mul},
        {"<built-in function sub>", op::translate_sub},
        {"aten._adaptive_avg_pool1d.default", op::translate_adaptive_avg_pool1d},
        {"aten._adaptive_avg_pool2d.default", op::translate_adaptive_avg_pool2d},
        {"aten._adaptive_avg_pool3d.default", op::translate_adaptive_avg_pool3d},
        {"aten._convolution.default", op::translate_convolution},
        {"aten._embedding_bag.default", op::translate_embedding_bag_fx},
        {"aten._fake_quantize_per_tensor_affine_cachemask_tensor_qparams.default",
         op::translate_fake_quantize_per_tensor_affine_fx},
        {"aten._local_scalar_dense.default", op::skip_node},
        {"aten._log_softmax.default", op::translate_log_softmax_fx},
        {"aten._native_batch_norm_legit.default", op::translate_batch_norm_legit_fx},
        {"aten._native_batch_norm_legit.no_stats", op::translate_batch_norm_legit_no_stats_fx},
        {"aten._native_batch_norm_legit_functional.default", op::translate_batch_norm_legit_fx},
        {"aten._native_batch_norm_legit_no_training.default", op::translate_batch_norm_legit_no_training_fx},
        {"aten._safe_softmax.default", op::translate_softmax_fx},
        {"aten._scaled_dot_product_flash_attention.default", op::translate_scaled_dot_product_attention_fx},
        {"aten._scaled_dot_product_flash_attention_for_cpu.default", op::translate_scaled_dot_product_attention_fx},
        {"aten._softmax.default", op::translate_softmax_fx},
        {"aten._to_copy.default", op::translate_to_fx},
        {"aten._unsafe_view.default", op::translate_reshape_fx},
        {"aten.abs.default", op::translate_1to1_match_1_inputs<opset10::Abs>},
        {"aten.acos.default", op::translate_1to1_match_1_inputs_with_fp32_type_alignment<opset10::Acos>},
        {"aten.acosh.default", op::translate_1to1_match_1_inputs_with_fp32_type_alignment<opset10::Acosh>},
        {"aten.adaptive_max_pool1d.default", op::translate_adaptive_max_pool1d_fx},
        {"aten.adaptive_max_pool2d.default", op::translate_adaptive_max_pool2d_fx},
        {"aten.adaptive_max_pool3d.default", op::translate_adaptive_max_pool3d_fx},
        {"aten.add.Scalar", op::translate_add},
        {"aten.add.Tensor", op::translate_add},
        {"aten.add_.Tensor", op::translate_add},
        {"aten.addcmul.default", op::translate_addcmul_fx},
        {"aten.addmm.default", op::translate_addmm_fx},
        {"aten.alias.default", op::skip_node},
        {"aten.all.default", op::translate_all},
        {"aten.amax.default", op::translate_amax},
        {"aten.amin.default", op::translate_amin},
        {"aten.any.default", op::translate_any_fx},
        {"aten.any.dim", op::translate_any_fx},
        {"aten.any.dims", op::translate_any_fx},
        {"aten.arange.default", op::translate_arange_fx},
        {"aten.arange.start", op::translate_arange_fx},
        {"aten.arange.start_step", op::translate_arange_fx},
        {"aten.argmax.default", op::translate_argmax},
        {"aten.argmin.default", op::translate_argmin},
        {"aten.as_strided.default", op::translate_as_strided},
        {"aten.as_strided_.default", op::translate_as_strided},
        {"aten.asin.default", op::translate_1to1_match_1_inputs_with_fp32_type_alignment<opset10::Asin>},
        {"aten.asinh.default", op::translate_1to1_match_1_inputs_with_fp32_type_alignment<opset10::Asinh>},
        {"aten.atan.default", op::translate_1to1_match_1_inputs_with_fp32_type_alignment<opset10::Atan>},
        {"aten.atanh.default", op::translate_1to1_match_1_inputs_with_fp32_type_alignment<opset10::Atanh>},
        {"aten.atan2.default", op::translate_atan2},
        {"aten.avg_pool2d.default", op::translate_avg_pool2d},
        {"aten.avg_pool3d.default", op::translate_avg_pool3d},
        {"aten.baddbmm.default", op::translate_addmm_fx},
        {"aten.bitwise_and.Scalar", op::translate_bitwise_and},
        {"aten.bitwise_and.Tensor", op::translate_bitwise_and},
        {"aten.bitwise_not.default", op::translate_bitwise_not},
        {"aten.bitwise_or.Tensor", op::translate_bitwise_or},
        {"aten.bitwise_xor.Tensor", op::translate_bitwise_xor},
        {"aten.bmm.default", op::translate_1to1_match_2_inputs_align_types<opset10::MatMul>},
        {"aten.cat.default", op::translate_cat_fx},
        {"aten.ceil.default", op::translate_1to1_match_1_inputs<opset10::Ceiling>},
        {"aten.celu.default", op::translate_celu},
        {"aten.clamp.default", op::translate_clamp},
        {"aten.clamp.Tensor", op::translate_clamp},
        {"aten.clamp_max.default", op::translate_1to1_match_2_inputs_align_types<opset10::Minimum>},
        {"aten.clamp_max.Tensor", op::translate_1to1_match_2_inputs_align_types<opset10::Minimum>},
        {"aten.clamp_min.default", op::translate_1to1_match_2_inputs_align_types<opset10::Maximum>},
        {"aten.clamp_min.Tensor", op::translate_1to1_match_2_inputs_align_types<opset10::Maximum>},
        {"aten.clone.default", op::skip_node},  // ignore clone operators that are inserted by PyTorch autograd
        {"aten.col2im.default", op::translate_col2im},
        {"aten.constant_pad_nd.default", op::translate_constant_pad_nd_fx},
        {"aten.convolution.default", op::translate_convolution},
        {"aten.copy.default", op::translate_copy_fx},
        {"aten.copy_.default", op::translate_copy_},
        {"aten.cos.default", op::translate_1to1_match_1_inputs_with_fp32_type_alignment<opset10::Cos>},
        {"aten.cosh.default", op::translate_1to1_match_1_inputs_with_fp32_type_alignment<opset10::Cosh>},
        {"aten.cumsum.default", op::translate_cumsum_fx},
        {"aten.channel_shuffle.default", op::translate_channel_shuffle},
        {"aten.detach.default", op::skip_node},
        {"aten.detach_.default", op::skip_node},
        {"aten.div.Scalar", op::translate_div_fx},
        {"aten.div.Tensor", op::translate_div_fx},
        {"aten.div.Tensor_mode", op::translate_div_fx},
        {"aten.div_.Tensor", op::translate_div_fx_},
        {"aten.elu.default", op::translate_elu},
        {"aten.elu_.default", op::inplace_op<op::translate_elu>},
        {"aten.embedding.default", op::translate_embedding},
        {"aten.empty.memory_format", op::translate_empty},
        {"aten.eq.Scalar", op::translate_1to1_match_2_inputs_align_types<opset10::Equal>},
        {"aten.eq.Tensor", op::translate_1to1_match_2_inputs_align_types<opset10::Equal>},
        {"aten.erf.default", op::translate_erf},
        {"aten.erfc.default", op::translate_erfc},
        {"aten.exp.default", op::translate_1to1_match_1_inputs_with_fp32_type_alignment<opset10::Exp>},
        {"aten.expm1.default", op::translate_expm1},
        {"aten.expand.default", op::translate_expand_fx},
        {"aten.eye.m", op::translate_eye_fx},
        {"aten.fake_quantize_per_channel_affine_cachemask.default", op::translate_fake_quantize_per_channel_affine_fx},
        {"aten.fill.Scalar", op::translate_fill},
        {"aten.fill_.Scalar", op::inplace_op<op::translate_fill>},
        {"aten.fill.Tensor", op::translate_fill},
        {"aten.fill_.Tensor", op::inplace_op<op::translate_fill>},
        {"aten.flip.default", op::translate_flip},
        {"aten.floor.default", op::translate_1to1_match_1_inputs<opset10::Floor>},
        {"aten.floor_divide.default", op::translate_floor_divide},
        {"aten.fmod.Scalar", op::translate_fmod},
        {"aten.fmod.Tensor", op::translate_fmod},
        {"aten.full.default", op::translate_full_fx},
        {"aten.full.names", op::translate_full_fx},
        {"aten.full_like.default", op::translate_full_like_fx},
        {"aten.gather.default", op::translate_gather},
        {"aten.ge.Scalar", op::translate_1to1_match_2_inputs_align_types<opset10::GreaterEqual>},
        {"aten.ge.Tensor", op::translate_1to1_match_2_inputs_align_types<opset10::GreaterEqual>},
        {"aten.gelu.default", op::translate_gelu_fx},
        {"aten.glu.default", op::translate_glu},
        {"aten.grid_sampler_2d.default", op::translate_grid_sampler},
        {"aten.gt.Scalar", op::translate_1to1_match_2_inputs_align_types<opset10::Greater>},
        {"aten.gt.Tensor", op::translate_1to1_match_2_inputs_align_types<opset10::Greater>},
        {"aten.hardsigmoid.default", op::translate_1to1_match_1_inputs<opset10::HSigmoid>},
        {"aten.hardswish.default", op::translate_1to1_match_1_inputs<opset10::HSwish>},
        {"aten.hardswish_.default", op::inplace_op<op::translate_1to1_match_1_inputs<opset10::HSwish>>},
        {"aten.hardtanh.default", op::translate_hardtanh},
        {"aten.hardtanh_.default", op::inplace_op<op::translate_hardtanh>},
        {"aten.index.Tensor", op::translate_index_fx},
        {"aten.index_select.default", op::translate_index_select},
        {"aten.isfinite.default", op::translate_1to1_match_1_inputs<opset10::IsFinite>},
        {"aten.isinf.default", op::translate_1to1_match_1_inputs<opset10::IsInf>},
        {"aten.isnan.default", op::translate_1to1_match_1_inputs<opset10::IsNaN>},
        {"aten.le.Scalar", op::translate_1to1_match_2_inputs_align_types<opset10::LessEqual>},
        {"aten.le.Tensor", op::translate_1to1_match_2_inputs_align_types<opset10::LessEqual>},
        {"aten.leaky_relu.default", op::translate_leaky_relu_fx},
        {"aten.leaky_relu_.default", op::inplace_op<op::translate_leaky_relu_fx>},
        {"aten.lift_fresh_copy.default", op::skip_node},
        {"aten.linalg_vector_norm.default", op::translate_linalg_vector_norm},
        {"aten.log.default", op::translate_1to1_match_1_inputs_with_fp32_type_alignment<opset10::Log>},
        {"aten.log_sigmoid_forward.default", op::translate_log_sigmoid_fx},
        {"aten.log10.default", op::translate_log10},
        {"aten.log1p.default", op::translate_log1p},
        {"aten.log2.default", op::translate_log2},
        {"aten.logical_and.default", op::translate_and},
        {"aten.logical_not.default", op::translate_not},
        {"aten.logsumexp.default", op::translate_logsumexp},
        {"aten.lt.Scalar", op::translate_1to1_match_2_inputs_align_types<opset10::Less>},
        {"aten.lt.Tensor", op::translate_1to1_match_2_inputs_align_types<opset10::Less>},
        {"aten.masked_fill.Scalar", op::translate_masked_fill},
        {"aten.masked_fill.Tensor", op::translate_masked_fill},
        {"aten.masked_fill_.Scalar", op::inplace_op<op::translate_masked_fill>},
        {"aten.masked_fill_.Tensor", op::inplace_op<op::translate_masked_fill>},
        {"aten.max.default", op::translate_max},
        {"aten.max.dim", op::translate_max_dim_fx},
        {"aten.max_pool2d_with_indices.default", op::translate_max_pool2d_fx},
        {"aten.max_pool3d_with_indices.default", op::translate_max_pool3d_fx},
        {"aten.maximum.default", op::translate_maximum},
        {"aten.mean.default", op::translate_mean_fx},
        {"aten.mean.dim", op::translate_mean_fx},
        {"aten.min.default", op::translate_min},
        {"aten.min.dim", op::translate_min_dim_fx},
        {"aten.minimum.default", op::translate_minimum},
        {"aten.mish.default", op::translate_1to1_match_1_inputs_with_fp32_type_alignment<opset10::Mish>},
        {"aten.mm.default", op::translate_1to1_match_2_inputs<opset10::MatMul>},
        {"aten.mul.Scalar", op::translate_mul},
        {"aten.mul.Tensor", op::translate_mul},
        {"aten.mul_.Tensor", op::translate_mul},
        {"aten.native_batch_norm.default", op::translate_batch_norm_legit_fx},
        {"aten.native_dropout.default", op::skip_node},
        {"aten.native_group_norm.default", op::translate_group_norm_fx},
        {"aten.native_layer_norm.default", op::translate_layer_norm_fx},
        {"aten.ne.Scalar", op::translate_1to1_match_2_inputs_align_types<opset10::NotEqual>},
        {"aten.ne.Tensor", op::translate_1to1_match_2_inputs_align_types<opset10::NotEqual>},
        {"aten.neg.default", op::translate_neg},
        {"aten.new_full.default", op::translate_new_full_fx},
        {"aten.new_ones.default", op::translate_new_ones_fx},
        {"aten.new_zeros.default", op::translate_new_zeros_fx},
        {"aten.ones.default", op::translate_ones_fx},
        {"aten.ones.names", op::translate_ones_fx},
        {"aten.ones_like.default", op::translate_ones_like_fx},
        {"aten.permute.default", op::translate_1to1_match_2_inputs<opset10::Transpose>},
        {"aten.pow.Scalar", op::translate_pow},
        {"aten.pow.Tensor_Scalar", op::translate_pow},
        {"aten.pow.Tensor_Tensor", op::translate_pow},
        {"aten.pixel_shuffle.default", op::translate_pixel_shuffle},
        {"aten.pixel_unshuffle.default", op::translate_pixel_unshuffle},
        {"aten.rand.default", op::translate_rand},
        {"aten.reciprocal.default", op::translate_reciprocal},
        {"aten.reflection_pad1d.default", op::translate_reflection_pad_nd_fx},
        {"aten.reflection_pad2d.default", op::translate_reflection_pad_nd_fx},
        {"aten.reflection_pad3d.default", op::translate_reflection_pad_nd_fx},
        {"aten.relu.default", op::translate_1to1_match_1_inputs<opset10::Relu>},
        {"aten.relu_.default", op::inplace_op<op::translate_1to1_match_1_inputs<opset10::Relu>>},
        {"aten.repeat.default", op::translate_1to1_match_2_inputs<opset10::Tile>},
        {"aten.roll.default", op::translate_roll},
        {"aten.rsqrt.default", op::translate_rsqrt},
        {"aten.rsub.Scalar", op::translate_rsub_fx},
        {"aten.rsub.Tensor", op::translate_rsub_fx},
        {"aten.scalar_tensor.default", op::translate_scalar_tensor_fx},
        {"aten.scatter.src", op::translate_scatter},
        {"aten.scatter.value", op::translate_scatter},
        {"aten.scatter_add.default", op::translate_scatter_add},
        {"aten.select.int", op::translate_select},
        {"aten.select_scatter.default", op::translate_select_scatter_fx},
        {"aten.sigmoid.default", op::translate_1to1_match_1_inputs_with_fp32_type_alignment<opset10::Sigmoid>},
        {"aten.sigmoid_.default", op::translate_1to1_match_1_inputs_with_fp32_type_alignment<opset10::Sigmoid>},
        {"aten.sign.default", op::translate_sign},
        {"aten.silu.default", op::translate_1to1_match_1_inputs<opset10::Swish>},
        {"aten.silu_.default", op::inplace_op<op::translate_1to1_match_1_inputs<opset10::Swish>>},
        {"aten.sin.default", op::translate_1to1_match_1_inputs_with_fp32_type_alignment<opset10::Sin>},
        {"aten.sinh.default", op::translate_1to1_match_1_inputs_with_fp32_type_alignment<opset10::Sinh>},
        {"aten.slice.Tensor", op::translate_slice_fx},
        {"aten.slice_scatter.default", op::translate_slice_scatter_fx},
        {"aten.sort.default", op::translate_sort_fx},
        {"aten.split.Tensor", op::translate_chunk_fx},
        {"aten.split_with_sizes.default", op::translate_split_with_sizes_fx},
        {"aten.sqrt.default", op::translate_1to1_match_1_inputs_with_fp32_type_alignment<opset10::Sqrt>},
        {"aten.squeeze.dim", op::translate_squeeze},
        {"aten.squeeze.dims", op::translate_squeeze},
        {"aten.stack.default", op::translate_stack_fx},
        {"aten.std.correction", op::translate_std_fx},
        {"aten.sub.default", op::translate_sub_fx},
        {"aten.sub.Tensor", op::translate_sub_fx},
        {"aten.sum.default", op::translate_sum_fx},
        {"aten.sum.dim_IntList", op::translate_sum_fx},
        {"aten.sym_size.int", op::translate_size},
        {"aten.t.default", op::translate_t},
        {"aten.tan.default", op::translate_1to1_match_1_inputs_with_fp32_type_alignment<opset10::Tan>},
        {"aten.tanh.default", op::translate_1to1_match_1_inputs_with_fp32_type_alignment<opset10::Tanh>},
        {"aten.topk.default", op::translate_topk_fx},
        {"aten.transpose.int", op::translate_transpose},
        {"aten.tril.default", op::translate_tril},
        {"aten.triu.default", op::translate_triu},
        {"aten.unbind.int", op::translate_unbind_int_fx},
        {"aten.unfold.default", op::translate_unfold},
        {"aten.unsqueeze.default", op::translate_1to1_match_2_inputs<opset10::Unsqueeze>},
        {"aten.upsample_nearest2d.default", op::translate_upsample_nearest2d},
        {"aten.var.correction", op::translate_var_fx},
        {"aten.var_mean.correction", op::translate_var_mean_fx},
        {"aten.view.default", op::translate_reshape_fx},
        {"aten.where.self", op::translate_where},
        {"aten.zeros.default", op::translate_zeros_fx},
        {"aten.zeros.names", op::translate_zeros_fx},
        {"aten.zeros_like.default", op::translate_zeros_like_fx},
        {"get_attr", op::translate_constant},
        {"torchvision.deform_conv2d.default", op::translate_deform_conv},
        {"torchvision.roi_align.default", op::translate_roi_align},
        {"quantized_decomposed.quantize_per_tensor.default", op::translate_quantize_per_tensor_fx},
        {"quantized_decomposed.quantize_per_channel.default", op::translate_quantize_per_channel_fx},
        {"quantized_decomposed.dequantize_per_tensor.default", op::skip_node},
        {"quantized_decomposed.dequantize_per_channel.default", op::skip_node},
    };
};

}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
