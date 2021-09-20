// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <op_table.hpp>
#include <default_opset.h>
#include <tensorflow_frontend/node_context.hpp>

using namespace std;
using namespace ngraph;
using namespace ngraph::frontend::tensorflow::detail;


#if 0

namespace tensorflow {
namespace ngraph_bridge {
// Translate SpaceToDepthOp
static Status TranslateSpaceToDepthOp(const TFNodeDecoder* op,
                                      const std::vector<const ngraph::frontend::tensorflow::detail::TensorWrapper*>&,
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
  auto ng_mode = opset::SpaceToDepth::SpaceToDepthMode::BLOCKS_FIRST;
  auto space_to_depth = ConstructNgNode<opset::SpaceToDepth>(
      node.get_name(), ng_input, ng_mode, block_size);
  NCHWtoNHWC(node.get_name(), is_nhwc, space_to_depth);
  SaveNgOp(ng_op_map, node.get_name(), space_to_depth);
  return Status::OK();
}
}
}

#endif