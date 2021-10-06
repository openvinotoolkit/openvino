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

static Status TranslatePackOp(const TFNodeDecoder* op, const std::vector<const ngraph::frontend::tf::detail::TensorWrapper*>&,
                              Builder::OpMap& ng_op_map) {
  TF_RETURN_IF_ERROR(ValidateInputCountMin(op, 1));

  int32_t tf_axis;
  TF_RETURN_IF_ERROR(GetNodeAttr(op->attrs(), "axis", &tf_axis));
  auto ng_axis = ConstructNgNode<Constant>(
      node.get_name(), element::i64, Shape{1},
      std::vector<int64_t>({tf_axis}));

  OutputVector ng_concat_inputs;
  for (int32_t i = 0; i < op->num_inputs(); ++i) {
    Output<Node> ng_input;
    TF_RETURN_IF_ERROR(GetInputNode(ng_op_map, op, i, ng_input));
    auto unsqueezed_input =
        ConstructNgNode<Unsqueeze>(node.get_name(), ng_input, ng_axis);
    ng_concat_inputs.push_back(unsqueezed_input);
  }

  // if inputs shape is (2, 3, 4), and axis is 1, then we want
  // to create output_shape (2, num_inputs, 3, 4)
  SaveNgOp(ng_op_map, node.get_name(), ConstructNgNode<Concat>(
                                      node.get_name(), ng_concat_inputs, tf_axis));
  return Status::OK();
}
}
}
#endif