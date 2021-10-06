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

OutputVector TranslateRangeOp(
    const NodeContext& node) {
  Output<Node> ng_start, ng_stop, ng_step;
  TF_RETURN_IF_ERROR(GetInputNodes(ng_op_map, op, ng_start, ng_stop, ng_step));

  //DataType start_type = op->input_type(0);
  //DataType stop_type = op->input_type(1);
  //DataType step_type = op->input_type(2);
  element::Type out_type;
  TF_RETURN_IF_ERROR(
      TFDataTypeToNGraphElementType(op->output_type(0), &out_type));
  //Output<Node> start_node, stop_node, step_node;
  //TF_RETURN_IF_ERROR(
  //    GetStaticInputNode(op, 0, static_input_map, start_type, start_node));
  //TF_RETURN_IF_ERROR(
  //    GetStaticInputNode(op, 1, static_input_map, stop_type, stop_node));
  //TF_RETURN_IF_ERROR(
  //    GetStaticInputNode(op, 2, static_input_map, step_type, step_node));
  auto ng_range = ConstructNgNode<Range>(node.get_name(), ng_start,
                                                ng_stop, ng_step, out_type);

  SaveNgOp(ng_op_map, node.get_name(), ng_range);
  return Status::OK();
}
}
}

#endif