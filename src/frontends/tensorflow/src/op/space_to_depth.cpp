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

OutputVector translate_space_to_depth_op(const NodeContext& node) {
    auto input = node.get_input(0);

    auto block_size = node.get_attribute<int64_t>("block_size");
    auto data_format = node.get_attribute<string>("data_format");

    TENSORFLOW_OP_VALIDATION(node, data_format == "NHWC" || data_format == "NCHW", "Unsupported data format.");

    bool is_nhwc = (data_format == "NHWC");
    convert_nhwc_to_nchw(node.get_name(), is_nhwc, input);
    auto ng_mode = SpaceToDepth::SpaceToDepthMode::BLOCKS_FIRST;
    auto res = make_shared<SpaceToDepth>(input, ng_mode, block_size)->output(0);
    convert_nchw_to_nhwc(node.get_name(), is_nhwc, res);
    set_node_name(node.get_name(), res.get_node_shared_ptr());
    return {res};
}
}  // namespace op
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
