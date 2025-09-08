// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "helper_ops/merge.hpp"

#include "common_op_table.hpp"
#include "helper_ops/enter.hpp"
#include "helper_ops/next_iteration.hpp"
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
    OutputVector inputs(input_size);
    for (int input_ind = 0; input_ind < input_size; ++input_ind) {
        inputs[input_ind] = node.get_input(input_ind);
    }

    // if Merge node has just one input, there is nothing to merge
    // return the same input and value_index equal to 0
    if (input_size == 1) {
        auto value_index = make_shared<v0::Constant>(element::i32, Shape{}, 0);
        value_index->output(0).set_names({node_name + ":1"});
        inputs[0].add_names({node_name + ":0"});
        return OutputVector{inputs[0], value_index};
    }

    // check if it is a case of TF1 While: Enter, NextIteration are going to Merge node
    // in this case it can refine output shape and type for NextIteration based on Enter
    if (input_size == 2) {
        auto enter = as_type_ptr<Enter>(inputs[0].get_node_shared_ptr());
        if (!enter) {
            enter = as_type_ptr<Enter>(inputs[1].get_node_shared_ptr());
        }
        auto next_iteration = as_type_ptr<NextIteration>(inputs[0].get_node_shared_ptr());
        if (!next_iteration) {
            next_iteration = as_type_ptr<NextIteration>(inputs[1].get_node_shared_ptr());
        }

        if (enter && next_iteration) {
            // set output type and shape for NextIteration
            // borrow them from Enter output
            auto enter_output_type = enter->output(0).get_element_type();
            auto enter_output_shape = enter->output(0).get_partial_shape();
            auto next_iteration_output_shape = PartialShape::dynamic(enter_output_shape.rank());
            next_iteration->set_output_shape_and_type(next_iteration_output_shape, enter_output_type);

            // reset inputs
            // refines input shapes and types for Merge node
            inputs[0] = enter->output(0);
            inputs[1] = next_iteration->output(0);
        }
    }

    auto merge_node = make_shared<Merge>(inputs, node.get_decoder());
    set_node_name(node.get_name(), merge_node);

    return merge_node->outputs();
}

}  // namespace op
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
