// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <default_opset.h>

#include <op_table.hpp>
#include <tensorflow_frontend/node_context.hpp>

using namespace std;
using namespace ngraph;
using namespace ngraph::frontend::tensorflow::detail;

#if 0

namespace tensorflow {
namespace ngraph_bridge {
static Status TranslateMatMulOp(const TFNodeDecoder* op,
                                const std::vector<const ngraph::frontend::tensorflow::detail::TensorWrapper*>&,
                                Builder::OpMap& ng_op_map) {
  Output<Node> ng_lhs, ng_rhs;
  TF_RETURN_IF_ERROR(GetInputNodes(ng_op_map, op, ng_lhs, ng_rhs));

  // Transpose arguments if requested.
  bool transpose_a = false;
  TF_RETURN_IF_ERROR(GetNodeAttr(op->attrs(), "transpose_a", &transpose_a));

  bool transpose_b = false;
  TF_RETURN_IF_ERROR(GetNodeAttr(op->attrs(), "transpose_b", &transpose_b));

  SaveNgOp(ng_op_map, node.get_name(),
           ConstructNgNode<opset::MatMul>(node.get_name(), ng_lhs, ng_rhs,
                                          transpose_a, transpose_b));
  return Status::OK();
}
}
}
#endif