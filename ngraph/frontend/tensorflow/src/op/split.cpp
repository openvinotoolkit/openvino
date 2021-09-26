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

OutputVector TranslateSplitOp(
    const NodeContext& node) {
  Output<Node> ng_input;
  TF_RETURN_IF_ERROR(GetInputNode(ng_op_map, op, 1, ng_input));
  // num_split : The number of ways to split. Must evenly divide
  // value.shape[split_dim]
  int32_t num_split;
  TF_RETURN_IF_ERROR(GetNodeAttr(op->attrs(), "num_split", &num_split));

  Shape shape = ng_input.get_shape();
  int rank = shape.size();

  std::vector<int> split_dim_vec;
  TF_RETURN_IF_ERROR(
      GetStaticInputVector(ng_op_map, op, 0, static_input_map, &split_dim_vec));
  int split_dim = split_dim_vec[0] + (split_dim_vec[0] < 0 ? (int64_t)rank : 0);
  auto ng_split_dim = ConstructNgNode<opset::Constant>(
      node.get_name(), element::u64, Shape{}, split_dim);
  auto ng_split = make_shared<opset::Split>(ng_input, ng_split_dim, num_split);

  for (int i = 0; i < num_split; ++i) {
    auto out = ng_split->output(i);
    Builder::SetTracingInfo(node.get_name(), out);
    SaveNgOp(ng_op_map, node.get_name(), out);
  }
  return Status::OK();
}

OutputVector TranslateSplitVOp(
    const NodeContext& node) {
  Output<Node> ng_input, ng_split_length, ng_split_dim;

  TF_RETURN_IF_ERROR(GetInputNode(ng_op_map, op, 0, ng_input));

  Shape shape = ng_input.get_shape();
  int rank = shape.size();

  std::vector<int64_t> split_dim_vec;
  TF_RETURN_IF_ERROR(
      GetStaticInputVector(ng_op_map, op, 2, static_input_map, &split_dim_vec));
  // there should be at least one element specified as axis and not more than
  // one as axis is 0-D
  if (split_dim_vec.size() != 1) {
    return errors::InvalidArgument(
        "split_dim_tensor must have "
        "exactly one element.");
  }
  TF_RETURN_IF_ERROR(CheckAxisDimInRange(split_dim_vec, rank));
  int split_dim = split_dim_vec[0] + (split_dim_vec[0] < 0 ? (int64_t)rank : 0);
  ng_split_dim = ConstructNgNode<opset::Constant>(node.get_name(), element::i32,
                                                  Shape{}, split_dim);

  std::vector<int> split_lengths_vec;
  TF_RETURN_IF_ERROR(
      GetStaticInputVector(ng_op_map, op, 1, static_input_map, &split_lengths_vec));

  // length: Length of size_splits
  int length = 0;
  int idx = -1;

  // Find out the total length of the splits and locate -1 's index, if any
  bool has_one_neg = false;
  for (size_t i = 0; i < split_lengths_vec.size(); ++i) {
    if (split_lengths_vec[i] != -1) {
      length += split_lengths_vec[i];
    } else {
      if (has_one_neg) {
        return errors::InvalidArgument("size_splits can only have one -1");
      } else {
        idx = i;
        has_one_neg = true;
      }
    }
  }

  // Size splits must sum to the dimension of value along split_dim
  if (idx > 0) {
    split_lengths_vec[idx] = shape[split_dim] - length;
  }

  if ((!has_one_neg && length != shape[split_dim]) ||
      (has_one_neg && split_lengths_vec[idx] < 0)) {
    return errors::InvalidArgument(
        "The length of size_splits must sum to the value of the dimension "
        "along split_dim");
  }

  ng_split_length = ConstructNgNode<opset::Constant>(
      node.get_name(), element::i32, Shape{split_lengths_vec.size()},
      split_lengths_vec);

  if (split_lengths_vec.size() != 1) {
    auto ng_split = make_shared<opset::VariadicSplit>(ng_input, ng_split_dim,
                                                      ng_split_length);
    for (size_t i = 0; i < split_lengths_vec.size(); ++i) {
      auto out = ng_split->output(i);
      Builder::SetTracingInfo(node.get_name(), out);
      SaveNgOp(ng_op_map, node.get_name(), out);
    }
  } else {
    SaveNgOp(ng_op_map, node.get_name(), ng_input);
  }

  return Status::OK();
}
}
}
#endif