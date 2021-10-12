// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <openvino/opsets/opset8.hpp>
#include <op_table.hpp>

using namespace std;
using namespace ov;
using namespace opset8;
using namespace ov::frontend;
using namespace frontend::tf::detail;

namespace ov {
namespace frontend {
namespace tf {
namespace op {

OutputVector TranslateConcatV2Op(const NodeContext& node) {
    ValidateInputCountMin(node, 2);

    std::vector<int64_t> tf_concat_axis_vec;
    GetStaticInputVector(node, node.get_ng_input_size() - 1, &tf_concat_axis_vec);

    int64_t concat_axis = tf_concat_axis_vec[0];

    if (concat_axis < 0) {
        auto ng_first_arg = node.get_ng_input(0);
        concat_axis += int64_t(ng_first_arg.get_shape().size());
    }

    OutputVector ng_args;

    for (int i = 0; i < node.get_ng_input_size() - 1; i++) {
        Output<Node> ng_arg = node.get_ng_input(i);
        ng_args.push_back(ng_arg);
    }

    return {ConstructNgNode<Concat>(node.get_name(), ng_args, size_t(concat_axis))};
}
}  // namespace op
}  // namespace tf
}  // namespace frontend
}  // namespace ngraph
