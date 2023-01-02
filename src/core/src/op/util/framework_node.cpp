// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/util/framework_node.hpp"

#include "itt.hpp"
#include "ngraph/graph_util.hpp"

ov::op::util::FrameworkNode::FrameworkNode(const OutputVector& inputs, size_t output_size, size_t num_subgraphs)
    : MultiSubGraphOp(inputs, num_subgraphs) {
    set_output_size(output_size);
    constructor_validate_and_infer_types();
}

ov::op::util::FrameworkNode::FrameworkNode(const ov::op::util::FrameworkNode& other) : MultiSubGraphOp() {
    set_arguments(other.input_values());
    other.clone_to(*this);
}

void ov::op::util::FrameworkNode::clone_to(ov::op::util::FrameworkNode& dst) const {
    dst.set_output_size(m_output_descriptions.size());

    for (size_t i = 0; i < get_output_size(); ++i) {
        dst.set_output_type(i, get_output_element_type(i), get_output_partial_shape(i));
    }
    dst.m_inputs_desc = m_inputs_desc;
    dst.m_output_desc = m_output_desc;
    dst.m_attrs = m_attrs;

    for (int i = 0; i < dst.m_bodies.size(); i++) {
        dst.m_bodies.push_back(ov::clone_model(*get_function(i)));
    }

    for (auto& input_description : m_input_descriptions[0]) {
        dst.m_input_descriptions[0].push_back(input_description->copy());
    }
    for (auto& output_description : m_output_descriptions[0]) {
        dst.m_output_descriptions[0].push_back(output_description->copy());
    }
    dst.validate_and_infer_types();
}

std::shared_ptr<ov::Node> ov::op::util::FrameworkNode::clone_with_new_inputs(const OutputVector& new_args) const {
    OV_OP_SCOPE(FrameworkNode_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    auto node = std::make_shared<op::util::FrameworkNode>(new_args);
    clone_to(*node);
    return node;
}

void ov::op::util::FrameworkNode::cache_output_descriptor() {
    for (size_t i = 0; i < get_output_size(); ++i) {
        m_output_desc.emplace_back(get_output_partial_shape(i), get_output_element_type(i));
    }
}

void ov::op::util::FrameworkNode::validate_and_infer_types() {
    OV_OP_SCOPE(FrameworkNode_validate_and_infer_types);
    // Save initial inputs descriptors
    bool initialize_input_desc = m_inputs_desc.empty();
    bool reset_output_shape_to_dynamic = false;
    bool reset_output_shape_to_original = false;
    for (uint64_t i = 0; i < get_input_size(); i++) {
        // TODO: store constant values
        const auto& input_pshape = get_input_partial_shape(i);
        const auto& input_type = get_input_element_type(i);
        const auto& rank = input_pshape.rank();

        const auto& get_error_message = [&]() {
            std::stringstream out;
            out << "Input descriptor for " << get_friendly_name() << " node has been changed:" << std::endl;
            out << "Before: " << std::get<0>(m_inputs_desc[i]) << ", " << std::get<1>(m_inputs_desc[i]) << std::endl;
            out << "After:  " << input_pshape << ", " << input_type << std::endl;
            out << "Please specify InferenceEngine Extensions to support this case.";
            return out.str();
        };

        if (initialize_input_desc) {
            m_inputs_desc.emplace_back(input_pshape, input_type);
        } else {
            const auto& orig_input_pshape = std::get<0>(m_inputs_desc[i]);
            if (orig_input_pshape == input_pshape) {
                reset_output_shape_to_original = true;
            } else if (input_pshape.rank().is_dynamic()) {
                reset_output_shape_to_dynamic = true;
            } else if (rank.is_static() && orig_input_pshape.rank().is_static() &&
                       rank.get_length() == orig_input_pshape.rank().get_length()) {
                for (int64_t dim = 0; dim < rank.get_length(); ++dim) {
                    NODE_VALIDATION_CHECK(this,
                                          input_pshape[dim].is_dynamic() ||
                                              (orig_input_pshape[dim].is_static() &&
                                               orig_input_pshape[dim].get_length() == input_pshape[dim].get_length()),
                                          get_error_message());
                }
                reset_output_shape_to_dynamic = true;
            } else {
                NODE_VALIDATION_CHECK(this,
                                      m_inputs_desc[i] == std::make_tuple(input_pshape, input_type),
                                      get_error_message());
            }
        }
    }

    if (reset_output_shape_to_dynamic) {
        cache_output_descriptor();
        for (size_t i = 0; i < get_output_size(); ++i) {
            if (get_output_partial_shape(i).rank().is_static()) {
                set_output_type(i, get_output_element_type(i), PartialShape::dynamic());
            }
        }
    }

    if (reset_output_shape_to_original && !m_output_desc.empty()) {
        for (size_t i = 0; i < get_output_size(); ++i) {
            set_output_type(i, std::get<1>(m_output_desc[i]), std::get<0>(m_output_desc[i]));
        }
    }
}

ov::AttributeAdapter<ov::op::util::FrameworkNodeAttrs>::AttributeAdapter(ov::op::util::FrameworkNodeAttrs& value)
    : DirectValueAccessor<ov::op::util::FrameworkNodeAttrs>(value) {}
