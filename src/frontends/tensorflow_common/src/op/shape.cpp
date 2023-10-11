// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_op_table.hpp"
#include "openvino/op/shape_of.hpp"

using namespace std;
using namespace ov;
using namespace ov::op;

namespace ov {
namespace frontend {
namespace tensorflow {
namespace op {

OutputVector translate_shape_op(const NodeContext& node) {
    default_op_checks(node, 1, {"Shape", "ShapeN", "SHAPE"});
    auto input_size = static_cast<int>(node.get_input_size());
    auto out_type = node.get_attribute<element::Type>("out_type", element::i32);
    auto node_name = node.get_name();

    if (input_size == 1) {
        auto input = node.get_input(0);
        auto shapeof = make_shared<v3::ShapeOf>(input, out_type);
        set_node_name(node_name, shapeof);
        return {shapeof};
    }

    OutputVector outputs;
    for (int input_ind = 0; input_ind < input_size; ++input_ind) {
        auto input = node.get_input(input_ind);
        auto shapeof = make_shared<v3::ShapeOf>(input, out_type);
        shapeof->set_friendly_name(node_name + "_" + to_string(input_ind));
        auto shapeof_output = shapeof->output(0);
        set_out_name({node_name + ":" + to_string(input_ind)}, shapeof_output);
        outputs.push_back(shapeof_output);
    }

    return outputs;
}

}  // namespace op
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
