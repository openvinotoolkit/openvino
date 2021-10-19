// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "op_table.hpp"

namespace ngraph {
namespace frontend {
namespace pdpd {
namespace op {
#define OP_CONVERTER(op) NamedOutputs op(const NodeContext& node)
OP_CONVERTER(argmax);
OP_CONVERTER(assign_value);
OP_CONVERTER(batch_norm);
OP_CONVERTER(bicubic_interp_v2);
OP_CONVERTER(bilinear_interp_v2);
OP_CONVERTER(cast);
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
OP_CONVERTER(gelu);
OP_CONVERTER(hard_sigmoid);
OP_CONVERTER(hard_swish);
OP_CONVERTER(layer_norm);
OP_CONVERTER(leaky_relu);
OP_CONVERTER(linear_interp_v2);
OP_CONVERTER(log);
OP_CONVERTER(logical_not);
OP_CONVERTER(matmul);
OP_CONVERTER(matmul_v2);
OP_CONVERTER(mul);
OP_CONVERTER(matrix_nms);
OP_CONVERTER(multiclass_nms);
OP_CONVERTER(nearest_interp_v2);
OP_CONVERTER(pad3d);
OP_CONVERTER(pow);
OP_CONVERTER(pool2d);
OP_CONVERTER(prior_box);
OP_CONVERTER(range);
OP_CONVERTER(relu);
OP_CONVERTER(relu6);
OP_CONVERTER(reshape2);
OP_CONVERTER(rnn);
OP_CONVERTER(scale);
OP_CONVERTER(shape);
OP_CONVERTER(slice);
OP_CONVERTER(softmax);
OP_CONVERTER(sigmoid);
OP_CONVERTER(split);
OP_CONVERTER(squeeze);
OP_CONVERTER(stack);
OP_CONVERTER(tanh);
OP_CONVERTER(transpose2);
OP_CONVERTER(trilinear_interp_v2);
OP_CONVERTER(unsqueeze);
OP_CONVERTER(yolo_box);
}  // namespace op
}  // namespace pdpd
}  // namespace frontend
}  // namespace ngraph

namespace ngraph {
namespace frontend {
namespace pdpd {
std::map<std::string, CreatorFunction> get_supported_ops() {
    return {{"arg_max", op::argmax},
            {"assign_value", op::assign_value},
            {"batch_norm", op::batch_norm},
            {"bicubic_interp_v2", op::bicubic_interp_v2},
            {"bilinear_interp_v2", op::bilinear_interp_v2},
            {"bilinear_interp", op::bilinear_interp_v2},
            {"bmm", op::matmul},
            {"cast", op::cast},
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
            {"gelu", op::gelu},
            {"greater_equal", op::elementwise_greater_equal},
            {"hard_sigmoid", op::hard_sigmoid},
            {"hard_swish", op::hard_swish},
            {"layer_norm", op::layer_norm},
            {"leaky_relu", op::leaky_relu},
            {"linear_interp_v2", op::linear_interp_v2},
            {"log", op::log},
            {"logical_not", op::logical_not},
            {"lookup_table_v2", op::embedding},
            {"matmul", op::matmul},
            {"matmul_v2", op::matmul_v2},
            {"max_pool2d_with_index", op::pool2d},
            {"mul", op::mul},
            {"matrix_nms", op::matrix_nms},
            {"multiclass_nms3", op::multiclass_nms},
            {"nearest_interp_v2", op::nearest_interp_v2},
            {"nearest_interp", op::nearest_interp_v2},
            {"pad3d", op::pad3d},
            {"pow", op::pow},
            {"pool2d", op::pool2d},
            {"prior_box", op::prior_box},
            {"range", op::range},
            {"relu", op::relu},
            {"relu6", op::relu6},
            {"reshape2", op::reshape2},
            {"rnn", op::rnn},
            {"scale", op::scale},
            {"shape", op::shape},
            {"slice", op::slice},
            {"softmax", op::softmax},
            {"sigmoid", op::sigmoid},
            {"split", op::split},
            {"squeeze2", op::squeeze},
            {"stack", op::stack},
            {"sync_batch_norm", op::batch_norm},
            {"tanh", op::tanh},
            {"transpose2", op::transpose2},
            {"trilinear_interp_v2", op::trilinear_interp_v2},
            {"unsqueeze2", op::unsqueeze},
            {"yolo_box", op::yolo_box}};
};

}  // namespace pdpd
}  // namespace frontend
}  // namespace ngraph