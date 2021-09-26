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

OutputVector TranslateIsFiniteOp(
    const NodeContext& node) {
  // Implemented tf.is_finite by checking:
  // (in != inf) && (in != -inf) && (in == in)
  //                                 ^^^^^^^^ checks for NaN's
  Output<Node> ng_input;
  TF_RETURN_IF_ERROR(GetInputNodes(ng_op_map, op, ng_input));

  auto const_inf = ConstructNgNode<opset::Constant>(
      node.get_name(), ng_input.get_element_type(), Shape{},
      std::vector<float>{std::numeric_limits<float>::infinity()});

  auto const_neg_inf = ConstructNgNode<opset::Constant>(
      node.get_name(), ng_input.get_element_type(), Shape{},
      std::vector<float>{-std::numeric_limits<float>::infinity()});

  auto neq_inf =
      ConstructNgNode<opset::NotEqual>(node.get_name(), ng_input, const_inf);
  auto neq_neg_inf =
      ConstructNgNode<opset::NotEqual>(node.get_name(), ng_input, const_neg_inf);
  auto eq_nan = ConstructNgNode<opset::Equal>(node.get_name(), ng_input, ng_input);

  auto neq_inf_and_neq_neg_inf =
      ConstructNgNode<opset::LogicalAnd>(node.get_name(), neq_inf, neq_neg_inf);
  auto is_finite = ConstructNgNode<opset::LogicalAnd>(
      node.get_name(), neq_inf_and_neq_neg_inf, eq_nan);

  SaveNgOp(ng_op_map, node.get_name(), is_finite);
  return Status::OK();
}
}
}
#endif