// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "template_function_transformation.hpp"

#include <openvino/cc/ngraph/itt.hpp>

using namespace ngraph;

// ! [function_pass:template_transformation_cpp]
// template_function_transformation.cpp
NGRAPH_RTTI_DEFINITION(ngraph::pass::MyFunctionTransformation, "MyFunctionTransformation", 0);

bool pass::MyFunctionTransformation::run_on_model(const std::shared_ptr<ov::Model>& f) {
    RUN_ON_FUNCTION_SCOPE(MyFunctionTransformation);
    // Example transformation code
    NodeVector nodes;

    // Traverse nGraph Function in topological order
    for (auto& node : f->get_ordered_ops()) {
        // Check that number of input and output ports are equal to 1
        if (node->inputs().size() == 1 && node->outputs().size() == 1) {
            // Check that input and output shape a fully defined (not dynamic) and number of consumers equal to 1
            Input<Node> input = node->input(0);
            Output<Node> output = node->output(0);
            if (input.get_partial_shape().is_static() && output.get_partial_shape().is_static() &&
                output.get_target_inputs().size() == 1) {
                nodes.push_back(node);
            }
        }
    }

    // Print types and names for collected nodes
    for (auto& node : nodes) {
        std::cout << "Type: " << node->get_type_info().name << std::endl
                  << "Name: " << node->get_friendly_name() << std::endl;
    }

    // Return false because we didn't change nGraph Function
    return false;
}
// ! [function_pass:template_transformation_cpp]
