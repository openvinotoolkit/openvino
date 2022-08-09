// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "op_table.hpp"
#include "openvino/opsets/opset8.hpp"
#include "utils.hpp"

using namespace std;
using namespace ov;
using namespace ov::frontend::tensorflow::detail;

namespace ov {
namespace frontend {
namespace tensorflow {
namespace op {

OutputVector translate_max_pool_op_util(const NodeContext& node, size_t spatial_dims_num) {
    TENSORFLOW_OP_VALIDATION(node, node.get_input_size() > 0, "MaxPool operation must have at least one input.");
    TENSORFLOW_OP_VALIDATION(node,
                             spatial_dims_num == 2 || spatial_dims_num == 3,
                             "Only MaxPool2D and MaxPool3D are supported.");
    auto input = node.get_input(0);

    // retrieve attributes
    auto tf_strides = node.get_attribute<std::vector<int64_t>>("strides");
    auto tf_ksize = node.get_attribute<std::vector<int64_t>>("ksize");
    auto tf_padding_type = node.get_attribute<std::string>("padding");
    ov::op::PadType auto_pad = convert_conv_tf_padding(node, tf_padding_type);
    auto tf_data_format = node.get_attribute<std::string>("data_format", spatial_dims_num == 2 ? "NHWC" : "NDHWC");

    auto tf_explicit_paddings = std::vector<int64_t>{};
    if (auto_pad == ov::op::PadType::EXPLICIT) {
        tf_explicit_paddings = node.get_attribute<std::vector<int64_t>>("explicit_paddings", {});
    }

    bool is_nhwc = true;
    if (spatial_dims_num == 2) {
        TENSORFLOW_OP_VALIDATION(node,
                                 tf_data_format == "NHWC" || tf_data_format == "NCHW",
                                 "MaxPool2D data format is neither NHWC nor NCHW");
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
    ov::frontend::tensorflow::convert_nhwc_to_hw(is_nhwc, tf_ksize, kernel_sizes);

    ov::CoordinateDiff pads_begin;
    ov::CoordinateDiff pads_end;
    if (auto_pad == ov::op::PadType::EXPLICIT) {
        fill_explicit_pads_vectors(node, is_nhwc, spatial_dims_num, tf_explicit_paddings, pads_begin, pads_end);
    }

    // prepare input to MaxPool
    convert_nhwc_to_nchw(is_nhwc, input);

    auto max_pool_node = std::make_shared<ov::opset8::MaxPool>(input,
                                                               strides,
                                                               dilations,
                                                               ov::Shape(pads_begin.begin(), pads_begin.end()),
                                                               ov::Shape(pads_end.begin(), pads_end.end()),
                                                               kernel_sizes,
                                                               ov::op::RoundingType::FLOOR,
                                                               auto_pad);
    auto max_pool = max_pool_node->output(0);
    ov::frontend::tensorflow::convert_nchw_to_nhwc(is_nhwc, max_pool);
    ov::frontend::tensorflow::set_node_name(node.get_name(), max_pool.get_node_shared_ptr());
    return {max_pool};
}

OutputVector translate_max_pool_op(const NodeContext& node) {
    if (node.get_op_type() == "MaxPool" || node.get_op_type() == "MaxPoolV2") {
        return translate_max_pool_op_util(node, 2);
    } else if (node.get_op_type() == "MaxPool3D") {
        return translate_max_pool_op_util(node, 3);
    } else {
        TENSORFLOW_OP_VALIDATION(node, false, "Only MaxPool2D and MaxPool3D are supported.");
    }
}

}  // namespace op
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov