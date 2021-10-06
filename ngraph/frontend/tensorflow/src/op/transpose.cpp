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

OutputVector TranslateTransposeOp(
    const NodeContext& node) {
  Output<Node> ng_input, ng_permutation;
  TF_RETURN_IF_ERROR(GetInputNodes(ng_op_map, op, ng_input, ng_permutation));
  SaveNgOp(ng_op_map, node.get_name(), ConstructNgNode<Transpose>(
                                      node.get_name(), ng_input, ng_permutation));
  return Status::OK();
}

}
}

#endif