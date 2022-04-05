// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "op_table.hpp"
#include "openvino/opsets/opset8.hpp"

using namespace std;
using namespace ov::opset8;

namespace ov {
namespace frontend {
namespace tensorflow {
namespace op {

OutputVector translate_lrn_op(const NodeContext& node) {
    auto input = node.get_input(0);
    auto alpha = node.get_attribute<float>("alpha");
    auto beta = node.get_attribute<float>("beta");
    auto bias = node.get_attribute<float>("bias");
    auto depth_radius = node.get_attribute<int64_t>("depth_radius");

    // OV: Each input value is divided by (bias+(alpha/size)*sum(xi^2 for every xi
    // in the local region))^beta
    // TF: sqr_sum[a, b, c, d] = sum(input[a, b, c, d - depth_radius : d +
    // depth_radius + 1] ** 2)
    //     output = input / (bias + alpha * sqr_sum) ** beta
    int64_t size = depth_radius * 2 + 1;
    alpha = alpha * size;
    // todo: input is in NHWC, need to apply NHWC to NCHW?
    auto res = make_shared<LRN>(input, alpha, beta, bias, static_cast<size_t>(size));
    set_node_name(node.get_name(), res);
    return res->outputs();
}

}  // namespace op
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov