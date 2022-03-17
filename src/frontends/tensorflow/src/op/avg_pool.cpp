// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "op_table.hpp"
#include "openvino/opsets/opset8.hpp"

using namespace std;
using namespace ov::opset8;

namespace ov {
namespace frontend {
namespace tensorflow {
namespace op {

OutputVector translate_avg_pool_op(const NodeContext& node) {
    Output<Node> ng_input = node.get_input(0);

    auto tf_strides = node.get_attribute<std::vector<int64_t>>("strides");
    auto tf_ksize = node.get_attribute<std::vector<int64_t>>("ksize");
    auto tf_padding_type = node.get_attribute<std::string>("padding");
    auto tf_data_format = node.get_attribute<std::string>("data_format");

    TENSORFLOW_OP_VALIDATION(
        node,
        tf_data_format == "NHWC" || tf_data_format == "NCHW" || tf_data_format == "NDHWC" || tf_data_format == "NCDHW",
        "AvgPool data format is neither NHWC (NDHWC) nor NCHW (NCDHW)");

    bool is_nhwc = (tf_data_format == "NHWC") || (tf_data_format == "NDHWC");

    int N = 2;
    if (node.get_op_type() == "AvgPool3D") {
        N = 3;
    }

    Strides ng_strides(N);
    Shape ng_image_shape(N);
    Shape ng_kernel_shape(N);
    convert_nhwc_to_hw(is_nhwc, tf_strides, ng_strides);
    convert_nhwc_to_hw(is_nhwc, ng_input.get_shape(), ng_image_shape);
    convert_nhwc_to_hw(is_nhwc, tf_ksize, ng_kernel_shape);
    convert_nhwc_to_nchw(node.get_name(), is_nhwc, ng_input);

    CoordinateDiff padding_below;
    CoordinateDiff padding_above;
    Shape ng_dilations(N, 1);
    make_padding(tf_padding_type,
                 ng_image_shape,
                 ng_kernel_shape,
                 ng_strides,
                 ng_dilations,
                 padding_below,
                 padding_above);

    // TODO: remove this once OV supports negative padding
    // (CoordinateDiff) for AvgPool
    Shape ng_padding_below(padding_below.begin(), padding_below.end());
    Shape ng_padding_above(padding_above.begin(), padding_above.end());

    auto res_node = make_shared<AvgPool>(ng_input,
                                         ng_strides,
                                         ng_padding_below,
                                         ng_padding_above,
                                         ng_kernel_shape,
                                         true,
                                         ov::op::RoundingType::FLOOR);
    auto res = res_node->output(0);

    convert_nchw_to_nhwc(node.get_name(), is_nhwc, res);
    set_node_name(node.get_name(), res.get_node_shared_ptr());
    return {res};
}
}  // namespace op
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov