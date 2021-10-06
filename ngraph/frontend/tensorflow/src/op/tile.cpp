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

OutputVector TranslateTileOp(
    const NodeContext& node) {
  Output<Node> ng_input, ng_multiples;
  TF_RETURN_IF_ERROR(GetInputNodes(ng_op_map, op, ng_input, ng_multiples));

  std::vector<int64_t> multiples;
  TF_RETURN_IF_ERROR(GetStaticInputVector(ng_op_map, op, 1, static_input_map, &multiples));

  auto ng_repeats = ConstructNgNode<Constant>(
      node.get_name(), element::i64, Shape{multiples.size()}, multiples);
  SaveNgOp(ng_op_map, node.get_name(),
           ConstructNgNode<Tile>(node.get_name(), ng_input, ng_repeats));
  return Status::OK();
}

}
}
#endif