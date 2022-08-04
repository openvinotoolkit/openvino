// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "op_table.hpp"
#include "openvino/opsets/opset8.hpp"

using namespace std;
using namespace ov;
using namespace opset8;
using namespace ov::frontend;
using namespace frontend::tensorflow::detail;

namespace ov {
namespace frontend {
namespace tensorflow {
namespace op {

OutputVector translate_concat_op(const NodeContext &node) {
  size_t axis_idx, concat_idx_start, concat_idx_stop;
  // Note that axis is going to different input ports to Concat and ConcatV2:
  // 0 - For Concat, N-1 - for ConcatV2
  if (node.get_op_type() == "ConcatV2") {
    axis_idx = node.get_input_size() - 1;
    concat_idx_start = 0;
    concat_idx_stop = node.get_input_size() - 1;
  } else if (node.get_op_type() == "Concat") {
    axis_idx = 0;
    concat_idx_start = 1;
    concat_idx_stop = node.get_input_size();
  } else {
    TENSORFLOW_OP_VALIDATION(node, false, "Incorrect operation type.");
  }

  std::vector<int64_t> tf_concat_axis_vec;
  get_const_input(node, axis_idx, &tf_concat_axis_vec);

  if (tf_concat_axis_vec.size() == 0) {
    TENSORFLOW_OP_VALIDATION(node, false, "Axis vector is empty.");
  }
  int64_t concat_axis = tf_concat_axis_vec[0];

  // Excluding empty input tensors
  OutputVector args;
  for (int i = concat_idx_start; i < concat_idx_stop; i++) {
    Output<Node> arg = node.get_input(i);
    bool is_empty_tensor = false;

    if (arg.get_partial_shape().rank().is_static()) {
      auto inp_shape = arg.get_partial_shape();
      for (auto dim : inp_shape) {
        if (dim.is_static() && dim == 0) {
          is_empty_tensor = true;
          break;
        }
      }
    }
    if (!is_empty_tensor) {
      args.push_back(arg);
    }
  }
  // Create a Const op if all inputs tensors are empty
  if (args.empty()) {
    int concat_axis_out_dim_value = 0;
    ov::Output<ov::Node> arg;
    ov::Shape inp_shape;
    for (int i = concat_idx_start; i < concat_idx_stop; i++) {
      Output<Node> arg = node.get_input(i);
      inp_shape = arg.get_shape();
      concat_axis_out_dim_value += inp_shape[concat_axis];
    }
    inp_shape[concat_axis] = concat_axis_out_dim_value;
    auto et = node.get_attribute<ov::element::Type>("T");
    auto res = make_shared<Constant>(et, inp_shape, 0);
    set_node_name(node.get_name(), res);
    return res->outputs();
  } else {
    auto res = make_shared<Concat>(args, int64_t(concat_axis));
    set_node_name(node.get_name(), res);
    return res->outputs();
  }
}
} // namespace op
} // namespace tensorflow
} // namespace frontend
} // namespace ov
