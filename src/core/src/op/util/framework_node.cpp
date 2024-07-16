// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/util/framework_node.hpp"

#include "itt.hpp"

ov::op::util::FrameworkNode::FrameworkNode(const OutputVector& inputs, size_t output_size, size_t num_subgraphs)
    : MultiSubGraphOp(num_subgraphs),
      m_num_bodies(num_subgraphs) {
    set_arguments(inputs);
    set_output_size(output_size);
    constructor_validate_and_infer_types();
}

ov::op::util::FrameworkNode::FrameworkNode(const ov::op::util::FrameworkNode& other) : MultiSubGraphOp() {
    set_arguments(other.input_values());
    other.clone_to(*this);
}

void ov::op::util::FrameworkNode::clone_to(ov::op::util::FrameworkNode& dst) const {
    dst.set_output_size(get_output_size());

    for (size_t i = 0; i < get_output_size(); ++i) {
        dst.set_output_type(i, get_output_element_type(i), get_output_partial_shape(i));
    }
    dst.m_inputs_desc = m_inputs_desc;
    dst.m_output_desc = m_output_desc;
    dst.m_attrs = m_attrs;
    dst.m_num_bodies = m_num_bodies;

    dst.m_bodies.resize(m_num_bodies);
    dst.m_input_descriptions.resize(m_num_bodies);
    dst.m_output_descriptions.resize(m_num_bodies);

    for (size_t i = 0; i < m_num_bodies; i++) {
        dst.m_bodies[i] = get_function(i)->clone();
        for (const auto& input_description : m_input_descriptions[i]) {
            dst.m_input_descriptions[i].push_back(input_description->copy());
        }
        for (auto& output_description : m_output_descriptions[i]) {
            dst.m_output_descriptions[i].push_back(output_description->copy());
        }
    }

    dst.validate_and_infer_types();
}

std::shared_ptr<ov::Node> ov::op::util::FrameworkNode::clone_with_new_inputs(const OutputVector& new_args) const {
    OV_OP_SCOPE(FrameworkNode_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    auto node = std::make_shared<op::util::FrameworkNode>(new_args, get_output_size(), m_num_bodies);
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

    if (m_inputs_desc.size() < get_input_size()) {
        // case when we added inputs using set_invariant_inputs
        m_inputs_desc.clear();
    }
    if (m_output_desc.size() < get_output_size()) {
        // case when we added outputs using set_body_outputs
        m_output_desc.clear();
    }

    // propagate shapes and types from bodies
    std::unordered_map<size_t, PartialShape> shape_map;
    std::unordered_map<size_t, element::Type> type_map;
    for (size_t i = 0; i < m_bodies.size(); ++i) {
        auto body = get_function(i);
        // If body doesn't exist skip the validation
        if (!body)
            continue;
        validate_and_infer_type_body(get_function(i), m_input_descriptions[i]);

        auto outputs_map = get_mapping_outputs_on_body_description(m_output_descriptions[i]);

        for (const auto& item : outputs_map) {
            auto output_index = item.first;
            auto desc = item.second;
            auto node_result = m_bodies[i]->get_results().at(desc->m_body_value_index)->input_value(0);
            auto pshape = PartialShape::dynamic();
            if (shape_map.count(output_index)) {
                pshape = shape_map.at(output_index);
            }
            if (PartialShape::merge_into(pshape, node_result.get_partial_shape())) {
                shape_map[output_index] = std::move(pshape);
            } else {
                shape_map[output_index] = PartialShape::dynamic();
            }
            auto type = element::dynamic;
            if (type_map.count(output_index)) {
                type = type_map.at(output_index);
            }
            if (element::Type::merge(type, type, node_result.get_element_type())) {
                type_map[output_index] = type;
            } else {
                type_map[output_index] = element::dynamic;
            }
        }
    }
    for (const auto& item : shape_map) {
        auto output_index = item.first;
        NODE_VALIDATION_CHECK(this,
                              type_map.count(output_index) != 0,
                              "Type map must contain same outputs as shape map");
        set_output_type(output_index, type_map.at(output_index), item.second);
    }

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
            out << "Please specify OpenVINO Extensions to support this case.";
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
                                          orig_input_pshape[dim].compatible(input_pshape[dim]),
                                          get_error_message());
                }
                reset_output_shape_to_dynamic = true;
            } else {
                NODE_VALIDATION_CHECK(
                    this,
                    orig_input_pshape.compatible(input_pshape) && std::get<1>(m_inputs_desc[i]).compatible(input_type),
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

bool ov::op::util::FrameworkNode::visit_attributes(AttributeVisitor& visitor) {
    visitor.on_attribute("framework_node_attrs", m_attrs);
    visitor.on_attribute("num_bodies", m_num_bodies);

    m_bodies.resize(m_num_bodies);
    m_input_descriptions.resize(m_num_bodies);
    m_output_descriptions.resize(m_num_bodies);

    for (size_t i = 0; i < m_num_bodies; ++i) {
        visitor.on_attribute("body" + std::to_string(i), m_bodies[i]);
        visitor.on_attribute("input_descriptions" + std::to_string(i), m_input_descriptions[i]);
        visitor.on_attribute("output_descriptions" + std::to_string(i), m_output_descriptions[i]);
    }
    return true;
}

ov::AttributeAdapter<ov::op::util::FrameworkNodeAttrs>::AttributeAdapter(ov::op::util::FrameworkNodeAttrs& value)
    : DirectValueAccessor<ov::op::util::FrameworkNodeAttrs>(value) {}
