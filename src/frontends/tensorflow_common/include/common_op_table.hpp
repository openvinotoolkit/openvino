// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <functional>
#include <map>
#include <string>

#include "openvino/core/node_vector.hpp"
#include "openvino/frontend/node_context.hpp"
#include "utils.hpp"

#define TENSORFLOW_OP_VALIDATION(node_context, ...)                                        \
    OPENVINO_ASSERT_HELPER(::ov::frontend::OpValidationFailure,                            \
                           ("While validating node '" + node_context.get_op_type() + "'"), \
                           __VA_ARGS__)

namespace ov {
namespace frontend {
namespace tensorflow {
namespace op {
#define OP_CONVERTER(op)       OutputVector op(const ov::frontend::NodeContext& node)
#define OP_CONVERTER_NAMED(op) NamedOutputVector op(const ov::frontend::NodeContext& node)
#define OP_T_CONVERTER(op) \
    template <class T>     \
    OutputVector op(const ov::frontend::NodeContext& node)

OP_T_CONVERTER(translate_unary_op);
OP_CONVERTER(translate_selu_op);
OP_T_CONVERTER(translate_binary_op);
OP_T_CONVERTER(translate_direct_reduce_op);
OP_CONVERTER(translate_addv2_op);
OP_CONVERTER(translate_add_n_op);
OP_CONVERTER(translate_approximate_equal_op);
OP_CONVERTER(translate_adjust_contrast_op);
OP_CONVERTER(translate_angle_op);
OP_CONVERTER(translate_arg_max_op);
OP_CONVERTER(translate_arg_min_op);
OP_CONVERTER(translate_atan2_op);
OP_CONVERTER(translate_avg_pool_op);
OP_CONVERTER(translate_batch_mat_mul_op);
OP_CONVERTER(translate_batch_mat_mul_with_type_op);
OP_CONVERTER(translate_batch_to_space_nd_op);
OP_CONVERTER(translate_bias_add_op);
OP_CONVERTER(translate_bincount_op);
OP_CONVERTER(translate_broadcast_args_op);
OP_CONVERTER(translate_broadcast_to_op);
OP_CONVERTER(translate_bucketize_op);
OP_CONVERTER(translate_cast_op);
OP_CONVERTER(translate_clip_by_value_op);
OP_CONVERTER(translate_complex_op);
OP_CONVERTER(translate_complex_abs_op);
OP_CONVERTER(translate_concat_op);
OP_CONVERTER(translate_conj_op);
OP_CONVERTER(translate_conj_transpose_op);
OP_CONVERTER(translate_const_op);
OP_CONVERTER(translate_conv_2d_op);
OP_CONVERTER(translate_conv_2d_backprop_input_op);
OP_CONVERTER(translate_conv_3d_op);
OP_CONVERTER(translate_conv_3d_backprop_input_v2_op);
OP_CONVERTER_NAMED(translate_ctc_greedy_decoder_op);
OP_CONVERTER(translate_ctc_loss_op);
OP_CONVERTER(translate_cumsum_op);
OP_CONVERTER(translate_crop_and_resize_op);
OP_CONVERTER(translate_depth_to_space_op);
OP_CONVERTER(translate_depthwise_conv_2d_native_op);
OP_CONVERTER(translate_div_no_nan_op);
OP_CONVERTER(translate_div_op);
OP_CONVERTER(translate_mul_op);
OP_CONVERTER(translate_dynamic_partition_op);
OP_CONVERTER(translate_einsum_op);
OP_CONVERTER(translate_elu_op);
OP_CONVERTER(translate_expand_dims_op);
OP_CONVERTER(translate_extract_image_patches_op);
OP_CONVERTER(translate_fake_quant_op);
OP_CONVERTER(translate_fake_quant_with_min_max_args);
OP_CONVERTER(translate_fft_op);
OP_CONVERTER(translate_fill_op);
OP_CONVERTER(translate_floor_div_op);
OP_CONVERTER_NAMED(translate_fused_batch_norm_op);
OP_CONVERTER(translate_gather_op);
OP_CONVERTER(translate_gather_v2_op);
OP_CONVERTER(translate_gather_nd_op);
OP_CONVERTER(translate_gather_tree_op);
OP_CONVERTER(translate_identity_op);
OP_CONVERTER(translate_identity_n_op);
OP_CONVERTER(translate_ifft_op);
OP_CONVERTER(translate_input_arg_op);
OP_CONVERTER(translate_inv_op);
OP_CONVERTER(translate_invert_permutation_op);
OP_CONVERTER(translate_output_arg_op);
OP_CONVERTER(translate_interpolate_op);
OP_CONVERTER(translate_irfft_op);
OP_CONVERTER(translate_is_finite_op);
OP_CONVERTER(translate_is_inf_op);
OP_CONVERTER(translate_is_nan_op);
OP_CONVERTER(translate_l2_loss_op);
OP_CONVERTER(translate_linspace_op);
OP_CONVERTER(translate_list_diff_op);
OP_CONVERTER(translate_leaky_relu_op);
OP_CONVERTER(translate_log_softmax_op);
OP_CONVERTER(translate_log_1p_op);
OP_CONVERTER(translate_lrn_op);
OP_CONVERTER(translate_mat_mul_op);
OP_CONVERTER(translate_matrix_diag_op);
OP_CONVERTER(translate_matrix_band_part_op);
OP_CONVERTER(translate_matrix_inverse_op);
OP_CONVERTER(translate_max_pool_op);
OP_CONVERTER_NAMED(translate_max_pool_with_argmax);
OP_CONVERTER(translate_mirror_pad_op);
OP_CONVERTER(translate_mul_no_nan_op);
OP_CONVERTER(translate_multinomial_op);
OP_CONVERTER_NAMED(translate_non_max_suppression_op);
OP_CONVERTER(translate_parallel_dynamic_stitch_op);
OP_CONVERTER(translate_placeholder_op);
OP_CONVERTER(translate_placeholder_with_default_op);
OP_CONVERTER(translate_no_op);
OP_CONVERTER(translate_one_hot_op);
OP_CONVERTER(translate_ones_like_op);
OP_CONVERTER(translate_pack_op);
OP_CONVERTER(translate_pad_op);
OP_CONVERTER(translate_padv2_op);
OP_CONVERTER(translate_range_op);
OP_CONVERTER(translate_rank_op);
OP_CONVERTER(translate_random_uniform_op);
OP_CONVERTER(translate_random_uniform_int_op);
OP_CONVERTER(translate_real_imag_op);
OP_CONVERTER(translate_relu_6_op);
OP_CONVERTER(translate_reciprocal_op);
OP_CONVERTER(translate_reshape_op);
OP_CONVERTER(translate_resource_gather_op);
OP_CONVERTER(translate_reverse_op);
OP_CONVERTER(translate_reverse_v2_op);
OP_CONVERTER(translate_reverse_sequence_op);
OP_CONVERTER(translate_rfft_op);
OP_CONVERTER(translate_roll_op);
OP_CONVERTER(translate_round_op);
OP_CONVERTER(translate_rsqrt_op);
OP_CONVERTER(translate_scatter_nd_op);
OP_CONVERTER(translate_segment_sum_op);
OP_CONVERTER(translate_space_to_batch_nd_op);
OP_CONVERTER(translate_sparse_to_dense_op);
OP_CONVERTER(translate_select_op);
OP_CONVERTER(translate_select_v2_op);
OP_CONVERTER(translate_shape_op);
OP_CONVERTER(translate_size_op);
OP_CONVERTER(translate_slice_op);
OP_CONVERTER(translate_softmax_op);
OP_CONVERTER(translate_space_to_depth_op);
OP_CONVERTER(translate_split_op);
OP_CONVERTER(translate_split_v_op);
OP_CONVERTER(translate_square_op);
OP_CONVERTER(translate_squeeze_op);
OP_CONVERTER(translate_strided_slice_op);
OP_CONVERTER(translate_sqrt_op);
OP_CONVERTER(translate_tensor_list_from_tensor_op);
OP_CONVERTER(translate_tensor_list_get_item_op);
OP_CONVERTER(translate_tensor_list_length_op);
OP_CONVERTER(translate_tensor_list_push_back_op);
OP_CONVERTER(translate_tensor_list_reserve_op);
OP_CONVERTER(translate_tensor_list_set_item_op);
OP_CONVERTER(translate_tensor_list_stack_op);
OP_CONVERTER(translate_tensor_list_resize_op);
OP_CONVERTER(translate_tensor_list_concat_v2_op);
OP_CONVERTER(translate_tile_op);
OP_CONVERTER(translate_tobool_op);
OP_CONVERTER_NAMED(translate_top_k_op);
OP_CONVERTER_NAMED(translate_top_k_v2_op);
OP_CONVERTER(translate_transpose_op);
OP_CONVERTER(translate_truncate_div_op);
OP_CONVERTER(translate_truncate_mod_op);
OP_CONVERTER(translate_unique_with_counts_op);
OP_CONVERTER(translate_unpack_op);
OP_CONVERTER(translate_unravel_index_op);
OP_CONVERTER(translate_unsorted_segment_sum_op);
OP_CONVERTER(translate_where_op);
OP_CONVERTER(translate_x_div_y_op);
OP_CONVERTER(translate_xlog1py_op);
OP_CONVERTER(translate_xlogy_op);
OP_CONVERTER(translate_zeros_like_op);

// Translators for internal operations
OP_CONVERTER_NAMED(translate_unique_op);

}  // namespace op
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
