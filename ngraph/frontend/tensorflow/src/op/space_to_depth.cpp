// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <ngraph/opsets/opset8.hpp>
#include <op_table.hpp>

using namespace std;
using namespace ngraph::opset8;

namespace ngraph {
namespace frontend {
namespace tf {
namespace op {

OutputVector TranslateSpaceToDepthOp(const NodeContext& node) {
    auto input = node.get_ng_input(0);

    auto block_size = node.get_attribute<int64_t>("block_size");
    auto data_format = node.get_attribute<string>("data_format");

    TF_OP_VALIDATION_CHECK(node, data_format != "NHWC" && data_format != "NCHW", "Unsupported data format.");

    bool is_nhwc = (data_format == "NHWC");
    NHWCtoNCHW(node.get_name(), is_nhwc, input);
    auto ng_mode = SpaceToDepth::SpaceToDepthMode::BLOCKS_FIRST;
    auto space_to_depth = make_shared<SpaceToDepth>(input, ng_mode, block_size)->output(0);
    NCHWtoNHWC(node.get_name(), is_nhwc, space_to_depth);
    space_to_depth.get_node_shared_ptr()->set_friendly_name(node.get_name());
    return {space_to_depth};
}
}  // namespace op
}  // namespace tf
}  // namespace frontend
}  // namespace ngraph
