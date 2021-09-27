// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <default_opset.h>

#include <op_table.hpp>

using namespace std;
using namespace ngraph;
using namespace ngraph::frontend::tensorflow::detail;

#if 0

namespace tensorflow {
namespace ngraph_bridge {

OutputVector TranslateTileOp(
    const NodeContext& node) {
  Output<Node> ng_input, ng_multiples;
  TF_RETURN_IF_ERROR(GetInputNodes(ng_op_map, op, ng_input, ng_multiples));

  std::vector<int64_t> multiples;
  TF_RETURN_IF_ERROR(GetStaticInputVector(ng_op_map, op, 1, static_input_map, &multiples));

  auto ng_repeats = ConstructNgNode<opset::Constant>(
      node.get_name(), element::i64, Shape{multiples.size()}, multiples);
  SaveNgOp(ng_op_map, node.get_name(),
           ConstructNgNode<opset::Tile>(node.get_name(), ng_input, ng_repeats));
  return Status::OK();
}

}
}
#endif