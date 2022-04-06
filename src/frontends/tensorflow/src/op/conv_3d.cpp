// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "op_table.hpp"
#include "openvino/opsets/opset8.hpp"

using namespace std;
using namespace ov::opset8;

// Translate Conv3D Op
namespace ov {
namespace frontend {
namespace tensorflow {
namespace op {

OutputVector translate_conv_3d_op(const NodeContext& node) {
    auto ng_input = node.get_input(0), ng_filter = node.get_input(1);

    auto tf_strides = node.get_attribute<std::vector<int64_t>>("strides");
    auto tf_dilations = node.get_attribute<std::vector<int64_t>>("dilations");
    auto tf_padding_type = node.get_attribute<std::string>("padding");
    auto tf_data_format = node.get_attribute<std::string>("data_format");

    TENSORFLOW_OP_VALIDATION(node,
                             tf_data_format == "NDHWC" || tf_data_format == "NCDHW",
                             "Conv3D data format is neither NDHWC nor NCDHW");

    bool is_ndhwc = (tf_data_format == "NDHWC");

    // TODO: in 3D
    // TF Kernel Test Checks
    // // Strides in the batch and depth dimension is not supported
    // if (tf_strides[0] != 1 || tf_strides[is_nhwc ? 3 : 1] != 1) {
    //   return errors::InvalidArgument(
    //       "Strides in batch and depth dimensions is not supported: ",
    //       op->type_string());
    // }

    Strides ng_strides(3);
    Strides ng_dilations(3);
    Shape ng_image_shape(3);
    Shape ng_kernel_shape(3);

    convert_nhwc_to_hw(is_ndhwc, tf_strides, ng_strides);
    convert_nhwc_to_hw(is_ndhwc, ng_input.get_shape(), ng_image_shape);
    convert_nhwc_to_hw(is_ndhwc, tf_dilations, ng_dilations);
    convert_nhwc_to_nchw(node.get_name(), is_ndhwc, ng_input);

    auto& ng_filter_shape = ng_filter.get_shape();
    ng_kernel_shape[0] = ng_filter_shape[0];
    ng_kernel_shape[1] = ng_filter_shape[1];
    ng_kernel_shape[2] = ng_filter_shape[2];
    transpose_3d<4, 3, 0, 1, 2>(ng_filter);

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
    auto res = res_node->output(0);

    convert_nchw_to_nhwc(node.get_name(), is_ndhwc, res);
    set_node_name(node.get_name(), res.get_node_shared_ptr());
    return {res};
}
}  // namespace op
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov