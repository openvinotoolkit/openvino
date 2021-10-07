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

OutputVector TranslateLRNOp(const NodeContext& node) {
    auto input = node.get_ng_input(0);
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
    auto lrn = make_shared<LRN>(input, alpha, beta, bias, static_cast<size_t>(size));
    lrn->set_friendly_name(node.get_name());
    return lrn->outputs();
}

}
}
}
}