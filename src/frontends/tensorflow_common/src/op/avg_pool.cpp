// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/avg_pool.hpp"

#include "common_op_table.hpp"

using namespace std;
using namespace ov::op;

namespace ov {
namespace frontend {
namespace tensorflow {
namespace op {

OutputVector translate_avg_pool_op(const NodeContext& node) {
    default_op_checks(node, 1, {"AvgPool", "AvgPool3D", "AVERAGE_POOL_2D"});
    auto op_type = node.get_op_type();
    auto input = node.get_input(0);

    auto spatial_dim = (op_type == "AvgPool" || op_type == "AVERAGE_POOL_2D") ? 2 : 3;

    // retrieve attributes for AvgPool operation
    auto tf_strides = node.get_attribute<std::vector<int64_t>>("strides");
    auto tf_ksize = node.get_attribute<std::vector<int64_t>>("ksize");
    auto tf_padding_type = node.get_attribute<std::string>("padding");
    ov::op::PadType auto_pad = convert_tf_padding(node, tf_padding_type);
    TENSORFLOW_OP_VALIDATION(node,
                             auto_pad == ov::op::PadType::VALID || auto_pad == ov::op::PadType::SAME_UPPER,
                             "AvgPool and AvgPool3D supports only VALID or SAME_UPPER padding mode.");

    // retrieve optional attribute
    auto tf_data_format = node.get_attribute<std::string>("data_format", (spatial_dim == 2) ? "NHWC" : "NDHWC");
    TENSORFLOW_OP_VALIDATION(
        node,
        tf_data_format == "NHWC" || tf_data_format == "NCHW" || tf_data_format == "NDHWC" || tf_data_format == "NCDHW",
        "AvgPool data format is neither NHWC (NDHWC) nor NCHW (NCDHW)");

    bool is_nhwc = (tf_data_format == "NHWC") || (tf_data_format == "NDHWC");

    // prepare inputs for OpenVINO AvgPool
    Strides strides(spatial_dim);
    Shape kernel_shape(spatial_dim);
    Shape dilations(spatial_dim, 1);
    convert_nhwc_to_hw(is_nhwc, tf_strides, strides);
    convert_nhwc_to_hw(is_nhwc, tf_ksize, kernel_shape);
    convert_nhwc_to_nchw(is_nhwc, input, ov::Rank(spatial_dim + 2));

    auto avg_pool = make_shared<v1::AvgPool>(input,
                                             strides,
                                             Shape({}),
                                             Shape({}),
                                             kernel_shape,
                                             true,
                                             ov::op::RoundingType::FLOOR,
                                             auto_pad);
    auto avg_pool_output = avg_pool->output(0);
    convert_nchw_to_nhwc(is_nhwc, avg_pool_output, ov::Rank(spatial_dim + 2));
    set_node_name(node.get_name(), avg_pool_output.get_node_shared_ptr());

    return {avg_pool_output};
}
}  // namespace op
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
