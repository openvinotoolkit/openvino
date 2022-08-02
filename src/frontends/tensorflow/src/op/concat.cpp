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

OutputVector translate_concat_op(const NodeContext& node) {
    size_t axis_idx, concat_idx_start, concat_idx_stop;
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
    int64_t concat_axis = tf_concat_axis_vec[0];

    // Excluding zero-dim inputs
    OutputVector ng_args;
    for (int i = concat_idx_start; i < concat_idx_stop; i++) {
        Output<Node> ng_arg = node.get_input(i);
        bool valid_input = true;

        if (ng_arg.get_partial_shape().is_static()) {
          auto inp_shape = ng_arg.get_shape();
          for (auto dim : inp_shape) {
            if (dim == 0) {
              valid_input = false;
              break;
            }
          }
        }
        if (valid_input) {
          ng_args.push_back(ng_arg);
        }
    }
    // Create a Const op if all inputs are zero-dim
    if (ng_args.empty()) {
      int concat_axis_out_dim_value = 0;
      ov::Output<ov::Node> ng_arg;
      ov::Shape inp_shape;
      for (int i = concat_idx_start; i < concat_idx_stop; i++) {
        Output<Node> ng_arg = node.get_input(i);
        inp_shape = ng_arg.get_shape();
        concat_axis_out_dim_value += inp_shape[concat_axis];
      }
      inp_shape[concat_axis] = concat_axis_out_dim_value;
      auto ng_et = node.get_attribute<ov::element::Type>("T");
      auto res = make_shared<Constant>(ng_et, inp_shape, 0);
      set_node_name(node.get_name(), res);
      return res->outputs();
    } else {
      auto res = make_shared<Concat>(ng_args, size_t(concat_axis));
      set_node_name(node.get_name(), res);
      return res->outputs();
    }
}
}  // namespace op
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
