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

OutputVector TranslateOneHotOp(
    const NodeContext& node) {
  Output<Node> ng_features, ng_unused, ng_on, ng_off, ng_depth;
  TF_RETURN_IF_ERROR(
      GetInputNodes(ng_op_map, op, ng_features, ng_unused, ng_on, ng_off));

  auto ng_features_shape = ng_features.get_shape();
  std::vector<int> depth;
  TF_RETURN_IF_ERROR(GetStaticInputVector(ng_op_map, op, 1, static_input_map, &depth));

  // Depth must be scalar
  if (depth.size() != 1) {
    return errors::InvalidArgument(
        "OneHot Op: depth of one hot dimension must be scalar " + to_string(depth.size()));
  }

  auto const_depth = ConstructNgNode<op::Constant>(
      node.get_name(), element::i64, Shape{}, depth);

  int one_hot_axis;
  TF_RETURN_IF_ERROR(GetNodeAttr(op->attrs(), "axis", &one_hot_axis));

  auto ng_onehot = ConstructNgNode<opset::OneHot>(
      node.get_name(), ng_features, const_depth, ng_on, ng_off, one_hot_axis);
  SaveNgOp(ng_op_map, node.get_name(), ng_onehot);
  return Status::OK();
}
}
}

#endif