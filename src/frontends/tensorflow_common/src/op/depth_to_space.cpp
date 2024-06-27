// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/depth_to_space.hpp"

#include "common_op_table.hpp"

using namespace std;
using namespace ov::op;

// Translate DepthToSpace op
namespace ov {
namespace frontend {
namespace tensorflow {
namespace op {

OutputVector translate_depth_to_space_op(const NodeContext& node) {
    default_op_checks(node, 1, {"DepthToSpace", "DEPTH_TO_SPACE"});
    auto input_data = node.get_input(0);

    // retrieve attributes
    auto block_size = node.get_attribute<int64_t>("block_size");
    auto data_format = node.get_attribute<string>("data_format", "NHWC");

    TENSORFLOW_OP_VALIDATION(node,
                             data_format == "NHWC" || data_format == "NCHW",
                             "TensorFlow Frontend supports input data for DepthToSpace either in NHWC or NCHW format.");

    bool is_nhwc = (data_format == "NHWC");

    convert_nhwc_to_nchw(is_nhwc, input_data);
    auto mode = v0::DepthToSpace::DepthToSpaceMode::BLOCKS_FIRST;
    auto depth_to_space = make_shared<v0::DepthToSpace>(input_data, mode, block_size)->output(0);
    convert_nchw_to_nhwc(is_nhwc, depth_to_space);
    set_node_name(node.get_name(), depth_to_space.get_node_shared_ptr());
    return {depth_to_space};
}
}  // namespace op
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
