// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <op_table.hpp>
#include <openvino/opsets/opset8.hpp>

using namespace std;
using namespace ov::opset8;

// Translate DepthToSpace op
namespace ov {
namespace frontend {
namespace tf {
namespace op {

OutputVector TranslateDepthToSpaceOp(const NodeContext& node) {
    Output<Node> ng_input = node.get_ng_input(0);

    // Get the attributes
    auto block_size = node.get_attribute<int64_t>("block_size");
    std::string tf_data_format = node.get_attribute<std::string>("data_format");

    if (tf_data_format != "NHWC" && tf_data_format != "NCHW") {
        throw errors::InvalidArgument("DepthToSpace data format is neither NHWC nor NCHW");
    }

    bool is_nhwc = (tf_data_format == "NHWC");

    NHWCtoNCHW(node.get_name(), is_nhwc, ng_input);
    auto ng_mode = DepthToSpace::DepthToSpaceMode::BLOCKS_FIRST;
    Output<Node> depth_to_space = ConstructNgNode<DepthToSpace>(node.get_name(), ng_input, ng_mode, block_size);
    NCHWtoNHWC(node.get_name(), is_nhwc, depth_to_space);
    return {depth_to_space};
}
}  // namespace op
}  // namespace tf
}  // namespace frontend
}  // namespace ov
