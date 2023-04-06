// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_op_table.hpp"
#include "openvino/opsets/opset8.hpp"
#include "utils.hpp"

using namespace std;
using namespace ov;
using namespace ov::frontend::tensorflow;

namespace ov {
namespace frontend {
namespace tensorflow {
namespace op {

OutputVector translate_max_pool_util(const NodeContext& node,
                                     size_t spatial_dims_num,
                                     const std::vector<int64_t>& tf_kernel_sizes,
                                     const std::vector<int64_t>& tf_strides) {
    default_op_checks(node, 1, {"MaxPool", "MaxPoolV2", "MaxPool3D"});
    TENSORFLOW_OP_VALIDATION(node,
                             spatial_dims_num == 2 || spatial_dims_num == 3,
                             "Only MaxPool, MaxPoolV2 and MaxPool3D are supported.");
    auto input = node.get_input(0);

    auto tf_padding_type = node.get_attribute<std::string>("padding");
    ov::op::PadType auto_pad = convert_tf_padding(node, tf_padding_type);
    auto tf_data_format = node.get_attribute<std::string>("data_format", spatial_dims_num == 2 ? "NHWC" : "NDHWC");

    auto tf_explicit_paddings = std::vector<int64_t>{};
    if (auto_pad == ov::op::PadType::EXPLICIT) {
        tf_explicit_paddings = node.get_attribute<std::vector<int64_t>>("explicit_paddings", {});
    }

    bool is_nhwc = true;
    if (spatial_dims_num == 2) {
        TENSORFLOW_OP_VALIDATION(node,
                                 tf_data_format == "NHWC" || tf_data_format == "NCHW",
                                 "MaxPool or MaxPoolV2 data format is neither NHWC nor NCHW");
        is_nhwc = (tf_data_format == "NHWC");
    } else {
        TENSORFLOW_OP_VALIDATION(node,
                                 tf_data_format == "NDHWC" || tf_data_format == "NCDHW",
                                 "MaxPool3D data format is neither NDHWC nor NCDHW");
        is_nhwc = (tf_data_format == "NDHWC");
    }

    // prepare attributes for OpenVINO MaxPool operation
    ov::Strides strides(spatial_dims_num);
    ov::Strides dilations = (spatial_dims_num == 2 ? ov::Strides({1, 1}) : ov::Strides({1, 1, 1}));
    ov::Shape kernel_sizes(spatial_dims_num);
    ov::frontend::tensorflow::convert_nhwc_to_hw(is_nhwc, tf_strides, strides);
    ov::frontend::tensorflow::convert_nhwc_to_hw(is_nhwc, tf_kernel_sizes, kernel_sizes);

    ov::CoordinateDiff pads_begin;
    ov::CoordinateDiff pads_end;
    if (auto_pad == ov::op::PadType::EXPLICIT) {
        fill_explicit_pads_vectors(node, is_nhwc, spatial_dims_num, tf_explicit_paddings, pads_begin, pads_end);
    }

    // prepare input to MaxPool
    convert_nhwc_to_nchw(is_nhwc, input, ov::Rank(spatial_dims_num + 2));

    auto max_pool_node = std::make_shared<ov::opset8::MaxPool>(input,
                                                               strides,
                                                               dilations,
                                                               ov::Shape(pads_begin.begin(), pads_begin.end()),
                                                               ov::Shape(pads_end.begin(), pads_end.end()),
                                                               kernel_sizes,
                                                               ov::op::RoundingType::FLOOR,
                                                               auto_pad);
    auto max_pool = max_pool_node->output(0);
    ov::frontend::tensorflow::convert_nchw_to_nhwc(is_nhwc, max_pool, ov::Rank(spatial_dims_num + 2));
    ov::frontend::tensorflow::set_node_name(node.get_name(), max_pool.get_node_shared_ptr());
    return {max_pool};
}

OutputVector translate_max_pool(const NodeContext& node, size_t spatial_dims_num) {
    // MaxPool2D and MaxPool3D have ksize and strides as attributes
    // retrieve attributes
    auto strides = node.get_attribute<std::vector<int64_t>>("strides");
    auto kernel_sizes = node.get_attribute<std::vector<int64_t>>("ksize");
    return translate_max_pool_util(node, spatial_dims_num, kernel_sizes, strides);
}

OutputVector translate_max_pool_v2(const NodeContext& node) {
    // MaxPoolV2 has ksize and strides as input parameters
    TENSORFLOW_OP_VALIDATION(node, node.get_input_size() > 2, "MaxPoolV2 operation must have at least three inputs.");
    auto ksize = node.get_input(1);
    auto strides = node.get_input(2);

    OPENVINO_SUPPRESS_DEPRECATED_START
    auto ksize_constant = get_constant_from_source(ksize);
    TENSORFLOW_OP_VALIDATION(node, ksize_constant, "MaxPoolV2 is supported only with constant ksize.");
    auto strides_constant = get_constant_from_source(strides);
    TENSORFLOW_OP_VALIDATION(node, ksize_constant, "MaxPoolV2 is supported only with constant strides.");
    OPENVINO_SUPPRESS_DEPRECATED_END

    auto ksize_vector = ksize_constant->cast_vector<int64_t>();
    auto strides_vector = strides_constant->cast_vector<int64_t>();

    return translate_max_pool_util(node, 2, ksize_vector, strides_vector);
}

OutputVector translate_max_pool_op(const NodeContext& node) {
    if (node.get_op_type() == "MaxPool") {
        return translate_max_pool(node, 2);
    } else if (node.get_op_type() == "MaxPoolV2") {
        return translate_max_pool_v2(node);
    } else if (node.get_op_type() == "MaxPool3D") {
        return translate_max_pool(node, 3);
    } else {
        TENSORFLOW_OP_VALIDATION(node, false, "Only MaxPool2D, MaxPoolV2 and MaxPool3D are supported.");
    }
}

}  // namespace op
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
