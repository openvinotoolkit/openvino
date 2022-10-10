// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "op_table.hpp"

namespace ov {
namespace frontend {
namespace paddle {
namespace op {
#define OP_CONVERTER(op) NamedOutputs op(const NodeContext& node)
OP_CONVERTER(argmax);
OP_CONVERTER(assign);
OP_CONVERTER(assign_value);
OP_CONVERTER(batch_norm);
OP_CONVERTER(bicubic_interp_v2);
OP_CONVERTER(bilinear_interp_v2);
OP_CONVERTER(cast);
OP_CONVERTER(ceil);
OP_CONVERTER(clip);
OP_CONVERTER(concat);
OP_CONVERTER(conv2d);
OP_CONVERTER(conv2d_transpose);
OP_CONVERTER(cumsum);
OP_CONVERTER(deformable_conv);
OP_CONVERTER(dropout);
OP_CONVERTER(elementwise_add);
OP_CONVERTER(elementwise_div);
OP_CONVERTER(elementwise_equal);
OP_CONVERTER(elementwise_greater_equal);
OP_CONVERTER(elementwise_max);
OP_CONVERTER(elementwise_min);
OP_CONVERTER(elementwise_mod);
OP_CONVERTER(elementwise_mul);
OP_CONVERTER(elementwise_pow);
OP_CONVERTER(elementwise_sub);
OP_CONVERTER(embedding);
OP_CONVERTER(exp);
OP_CONVERTER(expand_v2);
OP_CONVERTER(fill_any_like);
OP_CONVERTER(fill_constant_batch_size_like);
OP_CONVERTER(fill_constant);
OP_CONVERTER(flatten_contiguous_range);
OP_CONVERTER(floor);
OP_CONVERTER(gather);
OP_CONVERTER(gather_nd);
OP_CONVERTER(gelu);
OP_CONVERTER(greater_than);
OP_CONVERTER(group_norm);
OP_CONVERTER(hard_sigmoid);
OP_CONVERTER(hard_swish);
OP_CONVERTER(layer_norm);
OP_CONVERTER(leaky_relu);
OP_CONVERTER(less_than);
OP_CONVERTER(linear_interp_v2);
OP_CONVERTER(log);
OP_CONVERTER(logical_and);
OP_CONVERTER(logical_not);
OP_CONVERTER(logical_or);
OP_CONVERTER(logical_xor);
OP_CONVERTER(matmul);
OP_CONVERTER(matmul_v2);
OP_CONVERTER(matrix_nms);
OP_CONVERTER(meshgrid);
OP_CONVERTER(multiclass_nms);
OP_CONVERTER(nearest_interp_v2);
OP_CONVERTER(p_norm);
OP_CONVERTER(pad3d);
OP_CONVERTER(pow);
OP_CONVERTER(pool2d);
OP_CONVERTER(prior_box);
OP_CONVERTER(range);
OP_CONVERTER(reduce_max);
OP_CONVERTER(reduce_mean);
OP_CONVERTER(reduce_min);
OP_CONVERTER(reduce_prod);
OP_CONVERTER(reduce_sum);
OP_CONVERTER(relu);
OP_CONVERTER(relu6);
OP_CONVERTER(reshape2);
OP_CONVERTER(rnn);
OP_CONVERTER(roi_align);
OP_CONVERTER(scale);
OP_CONVERTER(shape);
OP_CONVERTER(slice);
OP_CONVERTER(softmax);
OP_CONVERTER(softplus);
OP_CONVERTER(sigmoid);
OP_CONVERTER(split);
OP_CONVERTER(sqrt);
OP_CONVERTER(squeeze);
OP_CONVERTER(stack);
OP_CONVERTER(strided_slice);
OP_CONVERTER(sum);
OP_CONVERTER(swish);
OP_CONVERTER(tanh);
OP_CONVERTER(tile);
OP_CONVERTER(top_k_v2);
OP_CONVERTER(transpose2);
OP_CONVERTER(trilinear_interp_v2);
OP_CONVERTER(unsqueeze);
OP_CONVERTER(where);
OP_CONVERTER(where_index);
OP_CONVERTER(yolo_box);
OP_CONVERTER(generate_proposals_v2);
}  // namespace op
std::map<std::string, CreatorFunction> get_supported_ops() {
    return {{"arg_max", op::argmax},
            {"assign", op::assign},
            {"assign_value", op::assign_value},
            {"batch_norm", op::batch_norm},
            {"bicubic_interp_v2", op::bicubic_interp_v2},
            {"bilinear_interp_v2", op::bilinear_interp_v2},
            {"bilinear_interp", op::bilinear_interp_v2},
            {"bmm", op::matmul},
            {"cast", op::cast},
            {"ceil", op::ceil},
            {"clip", op::clip},
            {"concat", op::concat},
            {"conv2d", op::conv2d},
            {"conv2d_transpose", op::conv2d_transpose},
            {"cumsum", op::cumsum},
            {"deformable_conv", op::deformable_conv},
            {"deformable_conv_v1", op::deformable_conv},
            {"depthwise_conv2d", op::conv2d},
            {"depthwise_conv2d_transpose", op::conv2d_transpose},
            {"dropout", op::dropout},
            {"elementwise_add", op::elementwise_add},
            {"elementwise_div", op::elementwise_div},
            {"elementwise_max", op::elementwise_max},
            {"elementwise_min", op::elementwise_min},
            {"elementwise_mod", op::elementwise_mod},
            {"elementwise_mul", op::elementwise_mul},
            {"elementwise_pow", op::elementwise_pow},
            {"elementwise_sub", op::elementwise_sub},
            {"equal", op::elementwise_equal},
            {"exp", op::exp},
            {"expand_v2", op::expand_v2},
            {"fill_any_like", op::fill_any_like},
            {"fill_constant_batch_size_like", op::fill_constant_batch_size_like},
            {"fill_constant", op::fill_constant},
            {"flatten_contiguous_range", op::flatten_contiguous_range},
            {"floor", op::floor},
            {"gather", op::gather},
            {"gather_nd", op::gather_nd},
            {"gelu", op::gelu},
            {"greater_equal", op::elementwise_greater_equal},
            {"greater_than", op::greater_than},
            {"group_norm", op::group_norm},
            {"hard_sigmoid", op::hard_sigmoid},
            {"hard_swish", op::hard_swish},
            {"layer_norm", op::layer_norm},
            {"leaky_relu", op::leaky_relu},
            {"less_than", op::less_than},
            {"linear_interp_v2", op::linear_interp_v2},
            {"log", op::log},
            {"logical_and", op::logical_and},
            {"logical_not", op::logical_not},
            {"logical_or", op::logical_or},
            {"logical_xor", op::logical_xor},
            {"lookup_table_v2", op::embedding},
            {"matmul", op::matmul},
            {"matmul_v2", op::matmul_v2},
            {"max_pool2d_with_index", op::pool2d},
            {"matrix_nms", op::matrix_nms},
            {"meshgrid", op::meshgrid},
            {"multiclass_nms3", op::multiclass_nms},
            {"nearest_interp_v2", op::nearest_interp_v2},
            {"nearest_interp", op::nearest_interp_v2},
            {"p_norm", op::p_norm},
            {"pad3d", op::pad3d},
            {"pow", op::pow},
            {"pool2d", op::pool2d},
            {"prior_box", op::prior_box},
            {"range", op::range},
            {"reduce_max", op::reduce_max},
            {"reduce_mean", op::reduce_mean},
            {"reduce_min", op::reduce_min},
            {"reduce_prod", op::reduce_prod},
            {"reduce_sum", op::reduce_sum},
            {"relu", op::relu},
            {"relu6", op::relu6},
            {"reshape2", op::reshape2},
            {"rnn", op::rnn},
            {"roi_align", op::roi_align},
            {"scale", op::scale},
            {"shape", op::shape},
            {"slice", op::slice},
            {"softmax", op::softmax},
            {"softplus", op::softplus},
            {"sigmoid", op::sigmoid},
            {"split", op::split},
            {"sqrt", op::sqrt},
            {"squeeze2", op::squeeze},
            {"stack", op::stack},
            {"strided_slice", op::strided_slice},
            {"sum", op::sum},
            {"swish", op::swish},
            {"sync_batch_norm", op::batch_norm},
            {"tanh", op::tanh},
            {"tile", op::tile},
            {"top_k_v2", op::top_k_v2},
            {"transpose2", op::transpose2},
            {"trilinear_interp_v2", op::trilinear_interp_v2},
            {"unsqueeze2", op::unsqueeze},
            {"where", op::where},
            {"where_index", op::where_index},
            {"yolo_box", op::yolo_box},
            {"generate_proposals_v2", op::generate_proposals_v2}};
};

}  // namespace paddle
}  // namespace frontend
}  // namespace ov
