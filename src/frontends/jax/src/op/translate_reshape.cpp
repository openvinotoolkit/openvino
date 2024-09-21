// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/jax/node_context.hpp"
#include "openvino/op/power.hpp"
#include "utils.hpp"

using namespace std;
using namespace ov;
using namespace ov::op;

namespace ov {
namespace frontend {
namespace jax {
namespace op {

OutputVector translate_gather(const NodeContext& context) {
    num_inputs_check(context, 2, 2); 
    Output<Node> input = context.get_input(0);
    Output<Node> indices = context.get_input(1);

    auto axis = context.const_named_param<int64_t>("axis");
    auto axis_node = std::make_shared<v0::Constant>(element::i64, Shape{}, axis);

    // the gather operation using OpenVINO Gather node
    Output<Node> res = std::make_shared<v1::Gather>(input, indices, axis_node);

    return {res};
}


} 
} 
} 
}  
