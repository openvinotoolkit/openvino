// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/pytorch/node_context.hpp"
#include "pt_framework_node.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

using namespace ov::op;

// aten::reverse.t(t[](a!) self) -> t[]
// Reverses the order of elements in a list
OutputVector translate_reverse(const NodeContext& context) {
    num_inputs_check(context, 1, 1);
    
    // Get the input list
    auto input = context.get_input(0);
    
    // Check if input is a list construct (has multiple outputs)
    auto input_node = input.get_node_shared_ptr();
    
    // For list inputs, we need to get all elements and reverse them
    if (auto list_node = std::dynamic_pointer_cast<PtFrameworkNode>(input_node)) {
        // If it's a framework node representing a list, get its inputs
        auto num_elements = list_node->get_input_size();
        OutputVector reversed;
        reversed.reserve(num_elements);
        
        // Reverse the order
        for (size_t i = num_elements; i > 0; --i) {
            reversed.push_back(list_node->input_value(i - 1));
        }
        
        return {context.mark_node(make_list_construct(reversed))};
    }
    
    // For single tensors or other cases, check if it has multiple inputs
    if (input_node->get_input_size() > 0) {
        auto num_elements = input_node->get_input_size();
        OutputVector reversed;
        reversed.reserve(num_elements);
        
        // Reverse the order of inputs
        for (size_t i = num_elements; i > 0; --i) {
            reversed.push_back(input_node->input_value(i - 1));
        }
        
        return {context.mark_node(make_list_construct(reversed))};
    }
    
    // If it's a single element, just return it as-is
    return {input};
}

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
