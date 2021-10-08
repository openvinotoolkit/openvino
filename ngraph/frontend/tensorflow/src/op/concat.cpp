// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <ngraph/opsets/opset8.hpp>
#include <op_table.hpp>

using namespace std;
using namespace ngraph;
using namespace opset8;
using namespace ngraph::frontend;
using namespace frontend::tf::detail;

namespace ngraph {
namespace frontend {
namespace tf {
namespace op {

OutputVector TranslateConcatOp(const NodeContext& node) {
    size_t axis_idx, concat_idx_start, concat_idx_stop;
    if (node.get_op_type() == "ConcatV2") {
        axis_idx = node.get_ng_input_size() - 1;
        concat_idx_start = 0;
        concat_idx_stop = node.get_ng_input_size() - 1;
    } else if (node.get_op_type() == "Concat") {
        axis_idx = 0;
        concat_idx_start = 1;
        concat_idx_stop = node.get_ng_input_size();
    } else {
        TF_OP_VALIDATION_CHECK(node, false, "Incorrect operation type.");
    }

    std::vector<int64_t> tf_concat_axis_vec;
    GetStaticInputVector(node, axis_idx, &tf_concat_axis_vec);
    int64_t concat_axis = tf_concat_axis_vec[0];

    OutputVector ng_args;
    for (int i = concat_idx_start; i < concat_idx_stop; i++) {
        Output<Node> ng_arg = node.get_ng_input(i);
        ng_args.push_back(ng_arg);
    }

    return {ConstructNgNode<Concat>(node.get_name(), ng_args, size_t(concat_axis))};
}
}  // namespace op
}  // namespace tf
}  // namespace frontend
}  // namespace ngraph