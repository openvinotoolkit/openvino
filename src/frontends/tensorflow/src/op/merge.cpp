// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "helper_ops/merge.hpp"

#include "common_op_table.hpp"
#include "openvino/frontend/tensorflow/node_context.hpp"
#include "openvino/op/constant.hpp"
#include "utils.hpp"

using namespace std;
using namespace ov;
using namespace ov::op;
using namespace ov::frontend::tensorflow;

namespace ov {
namespace frontend {
namespace tensorflow {
namespace op {

OutputVector translate_merge_op(const NodeContext& node) {
    // Merge can have multiple inputs, one is minimum
    auto node_name = node.get_name();
    default_op_checks(node, 1, {"Merge"});
    int input_size = static_cast<int>(node.get_input_size());
    OutputVector inputs;
    for (int input_ind = 0; input_ind < input_size; ++input_ind) {
        inputs.push_back(node.get_input(input_ind));
    }

    // if Merge node has just one input, there is nothing to merge
    // return the same input and value_index equal to 0
    if (inputs.size() == 1) {
        auto value_index = make_shared<v0::Constant>(element::i32, Shape{}, 0);
        value_index->output(0).set_names({node_name + ":1"});
        inputs[0].add_names({node_name + ":0"});
        return OutputVector{inputs[0], value_index};
    }

    auto merge_node = make_shared<Merge>(inputs, node.get_decoder());
    set_node_name(node.get_name(), merge_node);

    return merge_node->outputs();
}

}  // namespace op
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
