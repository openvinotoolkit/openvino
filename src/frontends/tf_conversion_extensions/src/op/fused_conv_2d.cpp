// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "conversion_extensions.hpp"
#include "openvino/frontend/tensorflow/node_context.hpp"
#include "openvino/opsets/opset8.hpp"

using namespace std;
using namespace ov::opset8;

namespace ov {
namespace frontend {
namespace tensorflow {
namespace op {

OutputVector translate_fused_conv_2d_op(const NodeContext& node) {
    auto num_args = node.get_attribute<int>("num_args");
    auto fused_ops = node.get_attribute<std::vector<string>>("fused_ops");

    auto tf_data_format = node.get_attribute<std::string>("data_format");
    bool is_nhwc = (tf_data_format == "NHWC");

    auto CreateNgConv = [&](Output<Node>& ng_input, Output<Node>& ng_filter) {
        auto tf_strides = node.get_attribute<std::vector<int64_t>>("strides");
        auto tf_dilations = node.get_attribute<std::vector<int64_t>>("dilations");
        auto tf_padding_type = node.get_attribute<std::string>("padding");

        if (tf_data_format != "NHWC" && tf_data_format != "NCHW") {
            FRONT_END_GENERAL_CHECK(false, "Conv2D data format is neither NHWC nor NCHW");
        }

        // TF Kernel Test Checks
        // Strides in the batch and depth dimension is not supported
        if (tf_strides[0] != 1 || tf_strides[is_nhwc ? 3 : 1] != 1) {
            FRONT_END_GENERAL_CHECK(false,
                                    "Strides in batch and depth dimensions is not supported: " + node.get_op_type());
        }

        Strides ng_strides(2);
        Strides ng_dilations(2);
        Shape ng_image_shape(2);
        Shape ng_kernel_shape(2);

        convert_nhwc_to_hw(is_nhwc, tf_strides, ng_strides);
        convert_nhwc_to_hw(is_nhwc, ng_input.get_shape(), ng_image_shape);
        convert_nhwc_to_hw(is_nhwc, tf_dilations, ng_dilations);
        convert_nhwc_to_nchw(node.get_name(), is_nhwc, ng_input);

        auto& ng_filter_shape = ng_filter.get_shape();
        ng_kernel_shape[0] = ng_filter_shape[0];
        ng_kernel_shape[1] = ng_filter_shape[1];
        Transpose<3, 2, 0, 1>(ng_filter);

        CoordinateDiff ng_padding_below;
        CoordinateDiff ng_padding_above;
        make_padding(tf_padding_type,
                     ng_image_shape,
                     ng_kernel_shape,
                     ng_strides,
                     ng_dilations,
                     ng_padding_below,
                     ng_padding_above);

        auto res_node =
            make_shared<Convolution>(ng_input, ng_filter, ng_strides, ng_padding_below, ng_padding_above, ng_dilations);
        return res_node->output(0);
    };

    if (vec_str_cmp(fused_ops, {"BiasAdd"}) || vec_str_cmp(fused_ops, {"BiasAdd", "Relu"}) ||
        vec_str_cmp(fused_ops, {"BiasAdd", "Relu6"})) {
        if (num_args != 1) {
            FRONT_END_GENERAL_CHECK(false, "FusedConv2DBiasAdd has incompatible num_args");
        }

        auto ng_input = node.get_input(0), ng_filter = node.get_input(1), ng_bias = node.get_input(2),
             ng_conv = CreateNgConv(ng_input, ng_filter);

        auto ng_conv_shape = ng_conv.get_shape();
        auto ng_bias_shape = ng_bias.get_shape();
        if (ng_bias_shape.size() != 1) {
            FRONT_END_GENERAL_CHECK(false, "Bias argument to BiasAdd does not have one dimension");
        }

        std::vector<size_t> reshape_pattern_values(ng_conv_shape.size(), 1U);
        reshape_pattern_values[1] = ng_bias.get_shape().front();
        auto reshape_pattern =
            make_shared<Constant>(element::u64, Shape{reshape_pattern_values.size()}, reshape_pattern_values);
        auto ng_bias_reshaped = make_shared<Reshape>(ng_bias, reshape_pattern, false);

        auto ng_add = make_shared<Add>(ng_conv, ng_bias_reshaped)->output(0);

        if (vec_str_cmp(fused_ops, {"BiasAdd", "Relu"})) {
            auto ng_relu = make_shared<Relu>(ng_add)->output(0);
            convert_nchw_to_nhwc(node.get_name(), is_nhwc, ng_relu);
            return {ng_relu};
        } else if (vec_str_cmp(fused_ops, {"BiasAdd", "Relu6"})) {
            auto ng_relu6 = make_shared<Clamp>(ng_add, 0, 6)->output(0);
            convert_nchw_to_nhwc(node.get_name(), is_nhwc, ng_relu6);
            return {ng_relu6};
        } else {
            convert_nchw_to_nhwc(node.get_name(), is_nhwc, ng_add);
            return {ng_add};
        }
    } else if (vec_str_cmp(fused_ops, {"FusedBatchNorm"}) || vec_str_cmp(fused_ops, {"FusedBatchNorm", "Relu"}) ||
               vec_str_cmp(fused_ops, {"FusedBatchNorm", "Relu6"})) {
        if (num_args != 4) {
            FRONT_END_GENERAL_CHECK(false, "FusedConv2D with FusedBatchNorm has incompatible num_args");
        }

        auto ng_input = node.get_input(0), ng_filter = node.get_input(1), ng_scale = node.get_input(2),
             ng_offset = node.get_input(3), ng_mean = node.get_input(4), ng_variance = node.get_input(5),
             ng_conv = CreateNgConv(ng_input, ng_filter);

        auto tf_epsilon = node.get_attribute<float>("epsilon");

        auto ng_batch_norm =
            make_shared<BatchNormInference>(ng_conv, ng_scale, ng_offset, ng_mean, ng_variance, tf_epsilon)->output(0);

        if (vec_str_cmp(fused_ops, {"FusedBatchNorm", "Relu"})) {
            auto ng_relu = make_shared<Relu>(ng_batch_norm)->output(0);
            convert_nchw_to_nhwc(node.get_name(), is_nhwc, ng_relu);
            return {ng_relu};
        } else if (vec_str_cmp(fused_ops, {"FusedBatchNorm", "Relu6"})) {
            auto ng_relu6 = make_shared<Clamp>(ng_batch_norm, 0, 6)->output(0);
            convert_nchw_to_nhwc(node.get_name(), is_nhwc, ng_relu6);
            return {ng_relu6};
        } else {
            convert_nchw_to_nhwc(node.get_name(), is_nhwc, ng_batch_norm);
            return {ng_batch_norm};
        }
    } else {
        FRONT_END_THROW("Unsupported _FusedConv2D ");
    }
}
}  // namespace op
}  // namespace tf
}  // namespace frontend
}  // namespace ov