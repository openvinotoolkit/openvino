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

static Status TranslateL2LossOp(const TFNodeDecoder* op,
                                const std::vector<const ngraph::frontend::tensorflow::detail::TensorWrapper*>&,
                                Builder::OpMap& ng_op_map) {
  Output<Node> ng_input;
  TF_RETURN_IF_ERROR(GetInputNodes(ng_op_map, op, ng_input));

  std::vector<float> val;
  val.push_back(2.0);
  auto const_2 = ConstructNgNode<opset::Constant>(
      node.get_name(), ng_input.get_element_type(), Shape{}, val[0]);

  auto ng_pow =
      ConstructNgNode<opset::Multiply>(node.get_name(), ng_input, ng_input);

  size_t input_rank = ng_input.get_shape().size();
  std::vector<int64_t> axes;
  for (size_t i = 0; i < input_rank; ++i) {
    axes.push_back(i);
  }

  auto ng_reduction_axes = ConstructNgNode<opset::Constant>(
      node.get_name(), element::i64, Shape{axes.size()}, axes);
  auto ng_sum =
      ConstructNgNode<opset::ReduceSum>(node.get_name(), ng_pow, ng_reduction_axes);
  auto ng_l2loss = ConstructNgNode<opset::Divide>(node.get_name(), ng_sum, const_2);
  SaveNgOp(ng_op_map, node.get_name(), ng_l2loss);
  return Status::OK();
}

}
}
#endif