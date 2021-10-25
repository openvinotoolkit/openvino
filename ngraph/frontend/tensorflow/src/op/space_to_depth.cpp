// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <op_table.hpp>
#include <openvino/opsets/opset8.hpp>

using namespace std;
using namespace ov::opset8;

namespace ov {
namespace frontend {
namespace tf {
namespace op {

OutputVector TranslateSpaceToDepthOp(const NodeContext& node) {
    auto input = node.get_ng_input(0);

    auto block_size = node.get_attribute<int64_t>("block_size");
    auto data_format = node.get_attribute<string>("data_format");

    TF_OP_VALIDATION_CHECK(node, data_format == "NHWC" || data_format == "NCHW", "Unsupported data format.");

    bool is_nhwc = (data_format == "NHWC");
    NHWCtoNCHW(node.get_name(), is_nhwc, input);
    auto ng_mode = SpaceToDepth::SpaceToDepthMode::BLOCKS_FIRST;
    auto res = make_shared<SpaceToDepth>(input, ng_mode, block_size)->output(0);
    NCHWtoNHWC(node.get_name(), is_nhwc, res);
    SetNodeNames(node.get_name(), res.get_node_shared_ptr());
    return {res};
}
}  // namespace op
}  // namespace tf
}  // namespace frontend
}  // namespace ov
