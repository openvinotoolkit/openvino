// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph_ops/framework_node.hpp"
#include "itt.hpp"

using namespace std;
using namespace ngraph;

NGRAPH_RTTI_DEFINITION(op::FrameworkNode, "FrameworkNode", 0);

op::FrameworkNode::FrameworkNode(const OutputVector& inputs)
    : Op(inputs) {
    constructor_validate_and_infer_types();
}

shared_ptr<Node> op::FrameworkNode::clone_with_new_inputs(const OutputVector& new_args) const {
    INTERNAL_OP_SCOPE(FrameworkNode_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    auto node = std::make_shared<op::FrameworkNode>(new_args);
    for (size_t i = 0; i < get_output_size(); ++i) {
        node->set_output_type(i, get_output_element_type(i), get_output_partial_shape(i));
    }
    return node;
}

void op::FrameworkNode::validate_and_infer_types() {
    INTERNAL_OP_SCOPE(FrameworkNode_validate_and_infer_types);
    // Save initial inputs descriptors
    bool initialize_input_desc = m_inputs_desc.empty();
    for (uint64_t i = 0; i < get_input_size(); i++) {
        // TODO: store constant values
        const auto& new_input_desc =
            std::make_tuple(get_input_partial_shape(i), get_input_element_type(i));

        if (initialize_input_desc) {
            m_inputs_desc.push_back(new_input_desc);
        } else {
            auto get_message = [&]() {
                std::stringstream out;
                out << "Input descriptor for " << get_friendly_name()
                    << " node has been changed:" << std::endl;
                out << "Before: " << std::get<0>(m_inputs_desc[i]) << ", "
                    << std::get<1>(m_inputs_desc[i]) << std::endl;
                out << "After:  " << std::get<0>(new_input_desc) << ", "
                    << std::get<1>(new_input_desc) << std::endl;
                out << "Please specify InferenceEngine Extensions to support this case.";
                return out.str();
            };

            NODE_VALIDATION_CHECK(this, m_inputs_desc[i] == new_input_desc, get_message());
        }
    }
}

constexpr DiscreteTypeInfo AttributeAdapter<op::FrameworkNodeAttrs>::type_info;

AttributeAdapter<op::FrameworkNodeAttrs>::AttributeAdapter(
    op::FrameworkNodeAttrs& value)
    : DirectValueAccessor<op::FrameworkNodeAttrs>(value) {}
