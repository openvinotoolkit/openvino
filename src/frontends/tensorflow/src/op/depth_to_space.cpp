// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "op_table.hpp"
#include "openvino/opsets/opset8.hpp"

using namespace std;
using namespace ov::opset8;

// Translate DepthToSpace op
namespace ov {
namespace frontend {
namespace tensorflow {
namespace op {

OutputVector translate_depth_to_space_op(const NodeContext& node) {
    Output<Node> ng_input = node.get_input(0);

    // Get the attributes
    auto block_size = node.get_attribute<int64_t>("block_size");
    std::string tf_data_format = node.get_attribute<std::string>("data_format");

    TENSORFLOW_OP_VALIDATION(node,
                             tf_data_format == "NHWC" || tf_data_format == "NCHW",
                             "DepthToSpace data format is neither NHWC nor NCHW");

    bool is_nhwc = (tf_data_format == "NHWC");

    convert_nhwc_to_nchw(node.get_name(), is_nhwc, ng_input);
    auto ng_mode = DepthToSpace::DepthToSpaceMode::BLOCKS_FIRST;
    Output<Node> res = make_shared<DepthToSpace>(ng_input, ng_mode, block_size)->output(0);
    convert_nchw_to_nhwc(node.get_name(), is_nhwc, res);
    set_node_name(node.get_name(), res.get_node_shared_ptr());
    return {res};
}
}  // namespace op
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov