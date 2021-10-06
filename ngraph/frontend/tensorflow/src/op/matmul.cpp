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
static Status TranslateMatMulOp(const TFNodeDecoder* op,
                                const std::vector<const ngraph::frontend::tf::detail::TensorWrapper*>&,
                                Builder::OpMap& ng_op_map) {
  Output<Node> ng_lhs, ng_rhs;
  TF_RETURN_IF_ERROR(GetInputNodes(ng_op_map, op, ng_lhs, ng_rhs));

  // Transpose arguments if requested.
  bool transpose_a = false;
  TF_RETURN_IF_ERROR(GetNodeAttr(op->attrs(), "transpose_a", &transpose_a));

  bool transpose_b = false;
  TF_RETURN_IF_ERROR(GetNodeAttr(op->attrs(), "transpose_b", &transpose_b));

  SaveNgOp(ng_op_map, node.get_name(),
           ConstructNgNode<MatMul>(node.get_name(), ng_lhs, ng_rhs,
                                          transpose_a, transpose_b));
  return Status::OK();
}
}
}
#endif