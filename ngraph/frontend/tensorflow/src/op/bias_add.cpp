// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <ngraph/opsets/opset8.hpp>
#include <op_table.hpp>

using namespace std;
using namespace ngraph::opset8;

namespace ngraph {
namespace frontend {
namespace tf {
namespace op {

OutputVector TranslateBiasAddOp(const NodeContext& node) {
    Output<Node> ng_input = node.get_ng_input(0), ng_bias = node.get_ng_input(1);

    std::string tf_data_format = node.get_attribute<std::string>("data_format", "NHWC");

    if (tf_data_format != "NHWC" && tf_data_format != "NCHW") {
        throw errors::InvalidArgument("BiasAdd data format is neither NHWC nor NCHW");
    }

    auto ng_input_shape = ng_input.get_shape();
    auto ng_bias_shape = ng_bias.get_shape();
    if (ng_bias_shape.size() != 1) {
        throw errors::InvalidArgument("Bias argument to BiasAdd does not have one dimension");
    }

    // We'll choose reshape over broadcast
    // Reshape the bias to (1, C, 1, ...) if input is channels-first.
    Output<Node> ng_bias_reshaped = ng_bias;
    if (tf_data_format == "NCHW") {
        auto channel_dim = ng_input_shape[1];
        std::vector<int64_t> target_shape(ng_input_shape.size());
        for (int64_t i = 0; i < ng_input_shape.size(); i++) {
            if (i == 1) {
                target_shape[i] = channel_dim;
            } else {
                target_shape[i] = 1;
            }
        }
        auto target_shape_node = make_shared<Constant>(element::i64, Shape{ng_input_shape.size()}, target_shape);
        ng_bias_reshaped = ConstructNgNode<Reshape>(node.get_name(), ng_bias, target_shape_node, false);
    }

    Output<Node> ng_add = ConstructNgNode<Add>(node.get_name(), ng_input, ng_bias_reshaped);

    return {ng_add};
}
}  // namespace op
}  // namespace tf
}  // namespace frontend
}  // namespace ngraph