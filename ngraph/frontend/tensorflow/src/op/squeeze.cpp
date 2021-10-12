// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <openvino/opsets/opset8.hpp>
#include <op_table.hpp>

using namespace std;
using namespace ov::opset8;

namespace ov {
namespace frontend {
namespace tf {
namespace op {

OutputVector TranslateSqueezeOp(const NodeContext& node) {
    Output<Node> ng_input = node.get_ng_input(0);
    size_t input_dims = ng_input.get_shape().size();

    auto tf_axis = node.get_attribute<std::vector<int32_t>>("squeeze_dims");

    // If input dimension is negative, make it positive
    for (size_t i = 0; i < tf_axis.size(); i++) {
        tf_axis[i] = tf_axis[i] < 0 ? (int32_t)(input_dims) + tf_axis[i] : tf_axis[i];
    }

    auto ng_const = ConstructNgNode<Constant>(node.get_name(), element::i32, Shape{tf_axis.size()}, tf_axis);

    return {ConstructNgNode<Squeeze>(node.get_name(), ng_input, ng_const)};
}

}  // namespace op
}  // namespace tf
}  // namespace frontend
}  // namespace ngraph
