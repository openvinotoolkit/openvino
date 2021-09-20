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

OutputVector TranslateTransposeOp(
    const NodeContext& node) {
  Output<Node> ng_input, ng_permutation;
  TF_RETURN_IF_ERROR(GetInputNodes(ng_op_map, op, ng_input, ng_permutation));
  SaveNgOp(ng_op_map, node.get_name(), ConstructNgNode<opset::Transpose>(
                                      node.get_name(), ng_input, ng_permutation));
  return Status::OK();
}

}
}

#endif