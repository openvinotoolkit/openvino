// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_op_table.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/mvn.hpp"
#include "openvino/op/power.hpp"
#include "openvino/op/range.hpp"
#include "openvino/op/reduce_mean.hpp"
#include "openvino/op/reduce_prod.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "utils.hpp"

using namespace std;
using namespace ov;
using namespace ov::op;

namespace ov {
namespace frontend {
namespace tensorflow {
namespace op {
namespace {
void generate_axes_range_except_c(const Output<Node>& x_rank, bool is_nhwc, Output<Node>& axes_no_c) {
    auto const_one = make_shared<v0::Constant>(element::i32, Shape{}, 1);
    if (is_nhwc) {
        auto const_zero = make_shared<v0::Constant>(element::i32, Shape{}, 0);
        auto rank_minus_one = make_shared<v1::Subtract>(x_rank, const_one);
        axes_no_c = make_shared<v4::Range>(const_zero, rank_minus_one, const_one, element::i32)->output(0);
    } else {
        auto const_zero = make_shared<v0::Constant>(element::i32, Shape{1}, 0);
        auto const_two = make_shared<v0::Constant>(element::i32, Shape{}, 2);
        // in NCHW layout case
        axes_no_c = make_shared<v4::Range>(const_two, x_rank, const_one, element::i32)->output(0);
        // add batch dimension as well
        axes_no_c = make_shared<v0::Concat>(OutputVector{const_zero, axes_no_c}, 0);
    }
}

void adjust_coeff(const Output<Node>& x_rank,
                  const Output<Node>& x,
                  const Output<Node>& coeff,
                  Output<Node>& adjusted_coeff,
                  bool is_nhwc) {
    // adjust types of the normalizing coefficients
    // they can vary for FusedBatchNormV2 and FusedBatchNormV3 operations
    adjusted_coeff = make_shared<v1::ConvertLike>(coeff, x)->output(0);

    if (is_nhwc) {
        return;
    }

    // in case NCHW format, we need to unsqueeze the normalizing coefficient by lower dimensions
    // to have the coefficient of shape [C, 1, 1]
    // generate axes range for unsqueezing the coefficient
    auto const_one = make_shared<v0::Constant>(element::i32, Shape{}, 1);
    auto x_rank_minus_one = make_shared<v1::Subtract>(x_rank, const_one);
    auto axes = make_shared<v4::Range>(const_one, x_rank_minus_one, const_one, element::i32);

    // adjust shapes of the normalizing coefficients
    adjusted_coeff = make_shared<v0::Unsqueeze>(adjusted_coeff, axes)->output(0);
}

void compute_batch_mean_and_variance(const Output<Node>& x,
                                     const Output<Node>& x_rank,
                                     bool is_nhwc,
                                     Output<Node>& batch_mean,
                                     Output<Node>& batch_variance) {
    // generate axes range for reduction operation
    Output<Node> reduce_axes;
    generate_axes_range_except_c(x_rank, is_nhwc, reduce_axes);

    // compute batch_mean
    batch_mean = make_shared<v1::ReduceMean>(x, reduce_axes, false)->output(0);

    // compute batch_variance
    auto unsqueezed_batch_mean = make_shared<v0::Unsqueeze>(batch_mean, reduce_axes);
    batch_variance = make_shared<v1::Subtract>(x, unsqueezed_batch_mean)->output(0);
    auto const_two = create_same_type_const_scalar<float>(x, 2);
    batch_variance = make_shared<v1::Power>(batch_variance, const_two);
    batch_variance = make_shared<v1::ReduceMean>(batch_variance, reduce_axes)->output(0);

    // for training mode, variance of FusedBatchNorm is computed with Bessel's correction
    // batch_variance must be multiplied by n / (n - 1), where n is a number of samples
    // to compute variance
    auto x_shape = make_shared<v3::ShapeOf>(x, element::i32);
    auto gather_axis = make_shared<v0::Constant>(element::i32, Shape{}, 0);
    auto needed_dim_values = make_shared<v8::Gather>(x_shape, reduce_axes, gather_axis);
    auto n = make_shared<v1::ReduceProd>(needed_dim_values, gather_axis, false)->output(0);
    n = make_shared<v1::ConvertLike>(n, batch_variance)->output(0);
    auto const_one = create_same_type_const_scalar<float>(batch_variance, 1);
    auto bessel_correction = make_shared<v1::Subtract>(n, const_one)->output(0);
    bessel_correction = make_shared<v1::Divide>(n, bessel_correction);

    // adjust batch_variance by bessel correction
    batch_variance = make_shared<v1::Multiply>(batch_variance, bessel_correction);
}

void compute_weighted_batch_mean_and_variance(const Output<Node>& x,
                                              const Output<Node>& mean,
                                              const Output<Node>& variance,
                                              const Output<Node>& exp_avg_factor_const,
                                              const Output<Node>& x_rank,
                                              const Output<Node>& batch_mean,
                                              const Output<Node>& batch_variance,
                                              Output<Node>& weighted_batch_mean,
                                              Output<Node>& weighted_batch_variance) {
    // compute weighted_mean and weighted_variance by the following formula:
    // (1 - exponential_avg_factor) * mean + exponential_avg_factor * batch_mean,
    // where batch_mean is the mean of the current batch in x.
    // for weighted_variance it is similar
    // (1 - exponential_avg_factor) * variance + exponential_avg_factor * batch_variance,
    // where batch_variance is the variance of the current batch in x.
    auto const_one = create_same_type_const_scalar<float>(exp_avg_factor_const, 1);
    auto one_minus_exp_avg_factor = make_shared<v1::Subtract>(const_one, exp_avg_factor_const);

    // compute weighted_batch_mean
    // no need to weight in case of empty tensor mean
    if (mean.get_partial_shape().is_static() && shape_size(mean.get_shape()) > 0) {
        auto bt_mean_by_exp_avg = make_shared<v1::Multiply>(batch_mean, exp_avg_factor_const);
        weighted_batch_mean = make_shared<v1::Multiply>(mean, one_minus_exp_avg_factor)->output(0);
        weighted_batch_mean = make_shared<v1::Add>(bt_mean_by_exp_avg, weighted_batch_mean);
    } else {
        weighted_batch_mean = batch_mean;
    }

    // compute weighted_batch_variance
    // no need to weight in case of empty tensor variance
    if (variance.get_partial_shape().is_static() && shape_size(variance.get_shape()) > 0) {
        auto bt_variance_by_exp_avg = make_shared<v1::Multiply>(batch_variance, exp_avg_factor_const);
        weighted_batch_variance = make_shared<v1::Multiply>(variance, one_minus_exp_avg_factor)->output(0);
        weighted_batch_variance = make_shared<v1::Add>(bt_variance_by_exp_avg, weighted_batch_variance)->output(0);
    } else {
        weighted_batch_variance = batch_variance;
    }
}

void compute_fused_batch_norm_inference(const NodeContext& node,
                                        Output<Node>& fused_batch_norm,
                                        Output<Node>& batch_mean,
                                        Output<Node>& batch_variance) {
    // when it is inference mode, there are five inputs: x, scale, offset, mean, and variance
    // The formula for FusedBatchNorm is the following:
    // (x - mean) / sqrt(variance + eps) * scale + offset
    default_op_checks(node, 5, {"FusedBatchNorm", "FusedBatchNormV2", "FusedBatchNormV3"});
    auto x = node.get_input(0);
    auto scale = node.get_input(1);
    auto offset = node.get_input(2);
    auto mean = node.get_input(3);
    auto variance = node.get_input(4);

    // retrieve attributes
    auto epsilon = node.get_attribute<float>("epsilon", 0.0001f);
    auto data_format = node.get_attribute<string>("data_format", "NHWC");
    bool is_nhwc = (data_format == "NHWC" || data_format == "NDHWC");

    // create auxiliary Constant nodes for some attributes: epsilon and exponential_avg_factor
    auto eps_const = create_same_type_const_scalar<float>(x, epsilon);
    auto half = create_same_type_const_scalar<float>(x, 0.5);

    // adjust normalizing coefficients: scale, offset, mean, and variance
    auto x_rank = compute_subgraph_scalar_rank(x, element::i32, true);
    Output<Node> adjusted_scale, adjusted_offset, adjusted_mean, adjusted_variance;
    adjust_coeff(x_rank, x, scale, adjusted_scale, is_nhwc);
    adjust_coeff(x_rank, x, offset, adjusted_offset, is_nhwc);
    adjust_coeff(x_rank, x, mean, adjusted_mean, is_nhwc);
    adjust_coeff(x_rank, x, variance, adjusted_variance, is_nhwc);

    // perform the main part of the transformation
    // 1. subtract mean from the input
    auto x_minus_mean = make_shared<v1::Subtract>(x, adjusted_mean);

    // 2. normalize the input after the shifting
    auto var_plus_eps = make_shared<v1::Add>(adjusted_variance, eps_const);
    auto root_sq_var = make_shared<v1::Power>(var_plus_eps, half);
    auto normalized_x = make_shared<v1::Divide>(x_minus_mean, root_sq_var);

    // 3. scale the input after the normalization
    auto scaled_x = make_shared<v1::Multiply>(normalized_x, adjusted_scale);
    fused_batch_norm = make_shared<v1::Add>(scaled_x, adjusted_offset)->output(0);

    // mean and variance go as outputs for batch_mean and batch_variance
    // exponential_avg_factor has no affect on it
    batch_mean = mean;
    batch_variance = variance;
}

void compute_fused_batch_norm_training(const NodeContext& node,
                                       Output<Node>& fused_batch_norm,
                                       Output<Node>& batch_mean,
                                       Output<Node>& batch_variance) {
    // when is_training is True, the operations have just three inputs: x, scale, and offset
    default_op_checks(node, 3, {"FusedBatchNorm", "FusedBatchNormV2", "FusedBatchNormV3"});
    auto x = node.get_input(0);
    auto scale = node.get_input(1);
    auto offset = node.get_input(2);

    // retrieve attributes
    auto epsilon = node.get_attribute<float>("epsilon", 0.0001f);
    auto data_format = node.get_attribute<string>("data_format", "NHWC");
    bool is_nhwc = (data_format == "NHWC");

    // adjust normalizing coefficients: scale, offset
    auto x_rank = compute_subgraph_scalar_rank(x, element::i32, true);
    Output<Node> adjusted_scale, adjusted_offset;
    adjust_coeff(x_rank, x, scale, adjusted_scale, is_nhwc);
    adjust_coeff(x_rank, x, offset, adjusted_offset, is_nhwc);

    // generate axes for MVN operations
    Output<Node> mvn_axes;
    generate_axes_range_except_c(x_rank, is_nhwc, mvn_axes);

    // perform mean-variance normalization
    auto mvn = make_shared<v6::MVN>(x, mvn_axes, true, epsilon, ov::op::MVNEpsMode::INSIDE_SQRT);

    // perform scaling and shifting
    fused_batch_norm = make_shared<v1::Multiply>(mvn, adjusted_scale)->output(0);
    fused_batch_norm = make_shared<v1::Add>(fused_batch_norm, adjusted_offset)->output(0);

    // compute two other outputs: batch_mean and batch_variance
    compute_batch_mean_and_variance(x, x_rank, is_nhwc, batch_mean, batch_variance);
    if (node.get_input_size() >= 5) {
        Output<Node> weighted_batch_mean, weighted_batch_variance;
        auto exponential_avg_factor = node.get_attribute<float>("exponential_avg_factor", 1.0f);
        auto exp_avg_factor_const = create_same_type_const_scalar<float>(scale, exponential_avg_factor);
        auto mean = node.get_input(3);
        auto variance = node.get_input(4);
        compute_weighted_batch_mean_and_variance(x,
                                                 mean,
                                                 variance,
                                                 exp_avg_factor_const,
                                                 x_rank,
                                                 batch_mean,
                                                 batch_variance,
                                                 weighted_batch_mean,
                                                 weighted_batch_variance);
        batch_mean = weighted_batch_mean;
        batch_variance = weighted_batch_variance;
    }
}
}  // namespace

NamedOutputVector translate_fused_batch_norm_op(const NodeContext& node) {
    default_op_checks(node, 3, {"FusedBatchNorm", "FusedBatchNormV2", "FusedBatchNormV3"});
    auto scale = node.get_input(1);

    // understand which version of the FusedBatchNorm operation
    auto op_type = node.get_op_type();
    auto is_v3 = (op_type == "FusedBatchNormV3");

    // there are two modes of FusedBatchNorm operations: training and inference
    // compute three meaningful outputs: fused_batch_norm, batch_mean and batch_variance
    Output<Node> fused_batch_norm, batch_mean, batch_variance;
    auto is_training = node.get_attribute<bool>("is_training", true);
    if (is_training) {
        compute_fused_batch_norm_training(node, fused_batch_norm, batch_mean, batch_variance);
    } else {
        compute_fused_batch_norm_inference(node, fused_batch_norm, batch_mean, batch_variance);
    }

    // create fictious output for reserved outputs of FusedBatchNorm operation
    auto zero_const = create_same_type_const_scalar<float>(scale, 0);
    auto zero_const2 = create_same_type_const_scalar<float>(scale, 0);

    // set node names and tensor names
    set_node_name(node.get_name(), fused_batch_norm.get_node_shared_ptr());
    set_out_name(node.get_name() + ":1", batch_mean.get_node_shared_ptr());
    set_out_name(node.get_name() + ":2", batch_variance.get_node_shared_ptr());
    set_out_name(node.get_name() + ":3", zero_const.get_node_shared_ptr());
    set_out_name(node.get_name() + ":4", zero_const2.get_node_shared_ptr());

    auto results = NamedOutputVector{{"y", fused_batch_norm},
                                     {"batch_mean", batch_mean},
                                     {"batch_variance", batch_variance},
                                     {"reserve_space_1", zero_const},
                                     {"reserve_space_2", zero_const2}};
    if (is_v3) {
        auto zero_const3 = create_same_type_const_scalar<float>(scale, 0);
        set_out_name(node.get_name() + ":5", zero_const3.get_node_shared_ptr());
        results.push_back({"reserve_space_3", zero_const3});
    }

    return results;
}
}  // namespace op
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
