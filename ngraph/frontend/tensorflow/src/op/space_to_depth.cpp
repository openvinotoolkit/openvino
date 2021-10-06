// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <ngraph/opsets/opset8.hpp>
#include <op_table.hpp>

using namespace std;
using namespace ngraph::opset8;

#if 0

namespace ngraph {
namespace frontend {
namespace tf {
namespace op {
// Translate SpaceToDepthOp
static Status TranslateSpaceToDepthOp(const TFNodeDecoder* op,
                                      const std::vector<const ngraph::frontend::tf::detail::TensorWrapper*>&,
                                      Builder::OpMap& ng_op_map) {
  Output<Node> ng_input;
  TF_RETURN_IF_ERROR(GetInputNodes(ng_op_map, op, ng_input));

  // Get the attributes
  int64_t block_size;
  std::string tf_data_format;
  TF_RETURN_IF_ERROR(GetNodeAttr(op->attrs(), "block_size", &block_size));
  TF_RETURN_IF_ERROR(GetNodeAttr(op->attrs(), "data_format", &tf_data_format));

  if (tf_data_format != "NHWC" && tf_data_format != "NCHW") {
    return errors::InvalidArgument(
        "DepthToSpace data format is neither NHWC nor NCHW");
  }

  bool is_nhwc = (tf_data_format == "NHWC");

  NHWCtoNCHW(node.get_name(), is_nhwc, ng_input);
  auto ng_mode = SpaceToDepth::SpaceToDepthMode::BLOCKS_FIRST;
  auto space_to_depth = ConstructNgNode<SpaceToDepth>(
      node.get_name(), ng_input, ng_mode, block_size);
  NCHWtoNHWC(node.get_name(), is_nhwc, space_to_depth);
  SaveNgOp(ng_op_map, node.get_name(), space_to_depth);
  return Status::OK();
}
}
}

#endif