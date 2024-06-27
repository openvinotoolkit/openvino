// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/if.hpp"

#include <algorithm>
#include <iterator>

#include "itt.hpp"
#include "openvino/core/graph_util.hpp"
#include "openvino/core/validation_util.hpp"
#include "openvino/op/util/multi_subgraph_base.hpp"
#include "openvino/reference/if.hpp"

ov::op::v8::If::If() : MultiSubGraphOp(2) {}

ov::op::v8::If::If(const Output<Node>& execution_condition) : If() {
    set_argument(0, execution_condition);
}

// This function tries to calculate the output shape of the if operation by two outputs from two
// subgraphs.
static ov::PartialShape resolve_shape(const ov::PartialShape& then_pshape, const ov::PartialShape& else_pshape) {
    // then_pshape - shape of output from then_body
    // else_pshape - shape of output from else_body
    auto then_rank = then_pshape.rank();
    auto else_rank = else_pshape.rank();

    // if rangs of shapes are not equal or rang of one of them is dynamic function
    // return shape with dynamic rank
    if (then_rank.is_dynamic() || else_rank.is_dynamic()) {
        return ov::PartialShape::dynamic();
    }
    if (then_rank.get_length() != else_rank.get_length()) {
        // Union of scalar and 1D case
        if (then_rank.get_length() <= 1 && else_rank.get_length() <= 1) {
            return ov::PartialShape::dynamic(1);
        } else {
            return ov::PartialShape::dynamic();
        }
    }
    ov::PartialShape new_dims;

    // If ranges are equal each dimension of then_body output is union with each dimension of
    // else_body
    for (auto then_it = then_pshape.cbegin(), else_it = else_pshape.cbegin(); then_it != then_pshape.cend();
         then_it++, else_it++) {
        if ((*then_it).is_dynamic() || (*else_it).is_dynamic()) {
            new_dims.push_back(ov::Dimension::dynamic());
        } else if (*then_it == *else_it) {
            new_dims.emplace_back(*then_it);
        } else {
            auto dim_min = std::min((*then_it).get_min_length(), (*else_it).get_min_length());
            auto dim_max = std::max((*then_it).get_min_length(), (*else_it).get_min_length());
            new_dims.emplace_back(dim_min, dim_max);
        }
    }

    return new_dims;
}

bool ov::op::v8::If::visit_attributes(AttributeVisitor& visitor) {
    OV_OP_SCOPE(v8_If_visit_attributes);
    visitor.on_attribute("then_body", m_bodies[THEN_BODY_INDEX]);
    visitor.on_attribute("then_inputs", m_input_descriptions[THEN_BODY_INDEX]);
    visitor.on_attribute("then_outputs", m_output_descriptions[THEN_BODY_INDEX]);
    visitor.on_attribute("else_body", m_bodies[ELSE_BODY_INDEX]);
    visitor.on_attribute("else_inputs", m_input_descriptions[ELSE_BODY_INDEX]);
    visitor.on_attribute("else_outputs", m_output_descriptions[ELSE_BODY_INDEX]);
    return true;
}

void ov::op::v8::If::validate_and_infer_types() {
    OV_OP_SCOPE(v8_If_validate_and_infer_types);

    NODE_VALIDATION_CHECK(this, m_bodies.size() == 2, "If contains incorrect number of bodies:", m_bodies.size());

    NODE_VALIDATION_CHECK(this,
                          m_input_descriptions.size() == 2,
                          "If contains incorrect number of body input descriptions:",
                          m_input_descriptions.size());
    NODE_VALIDATION_CHECK(this,
                          m_output_descriptions.size() == 2,
                          "If contains incorrect number of body output descriptions:",
                          m_output_descriptions.size());

    const auto& if_condition = input_value(0);
    const auto& if_condition_rank = if_condition.get_partial_shape().rank();
    if (if_condition_rank.is_static()) {
        NODE_VALIDATION_CHECK(this,
                              if_condition_rank.compatible(1) || if_condition_rank.compatible(0),
                              "Rank of If condition input must be equal to 0 or 1");
    }

    // Trying to get cond as const value
    if (const auto cond_value = ov::util::get_constant_from_source(if_condition)) {
        // If cond is const shape and inference is run for one of bodies another body is skipped
        auto val = cond_value->cast_vector<bool>();
        NODE_VALIDATION_CHECK(this,
                              val.size() == 1,
                              "The number of values in the If condition constant is greater than 1");

        // Condition is constant we only need to validate one body
        if (val[0]) {
            validate_and_infer_type_body(get_then_body(), m_input_descriptions[THEN_BODY_INDEX]);
        } else {
            validate_and_infer_type_body(get_else_body(), m_input_descriptions[ELSE_BODY_INDEX]);
        }
        auto output_nodes = outputs();

        auto cond_index = val[0] ? THEN_BODY_INDEX : ELSE_BODY_INDEX;
        auto body = m_bodies[cond_index];
        // shape and type inference for outputs from If operations
        for (const auto& output_descr : m_output_descriptions[cond_index]) {
            auto body_value = body->get_results().at(output_descr->m_body_value_index)->input_value(0);
            const auto& body_value_partial_shape = body_value.get_partial_shape();
            set_output_type(output_descr->m_output_index, body_value.get_element_type(), body_value_partial_shape);
        }
    } else  // condition is non constant
    {
        // If cond is non const, shape and type inference is run for both bodies
        validate_and_infer_type_body(get_then_body(), m_input_descriptions[THEN_BODY_INDEX]);
        validate_and_infer_type_body(get_else_body(), m_input_descriptions[ELSE_BODY_INDEX]);
        auto output_nodes = outputs();

        // Getting map<output_index_from_if, output_description>. This map guarantees that each
        // output from the body will be met in it once.
        auto then_outputs_map = get_mapping_outputs_on_body_description(m_output_descriptions[THEN_BODY_INDEX]);
        auto else_outputs_map = get_mapping_outputs_on_body_description(m_output_descriptions[ELSE_BODY_INDEX]);

        // Checking each output from If. Each output must be associated with one output from each
        // body
        for (size_t output_index = 0; output_index < output_nodes.size(); ++output_index) {
            NODE_VALIDATION_CHECK(this,
                                  then_outputs_map.count(output_index) != 0,
                                  "Incorrect associating in then_body! Output ",
                                  output_index,
                                  " is not associated with results in then_body!");
            NODE_VALIDATION_CHECK(this,
                                  else_outputs_map.count(output_index) != 0,
                                  "Incorrect associating in else_body! Output ",
                                  output_index,
                                  " is not associated with results in else_body!");

            auto then_desc = then_outputs_map.at(output_index);
            auto else_desc = else_outputs_map.at(output_index);

            auto then_node_result =
                m_bodies[THEN_BODY_INDEX]->get_results().at(then_desc->m_body_value_index)->input_value(0);

            auto else_node_result =
                m_bodies[ELSE_BODY_INDEX]->get_results().at(else_desc->m_body_value_index)->input_value(0);

            element::Type merged_type;
            NODE_VALIDATION_CHECK(this,
                                  element::Type::merge(merged_type,
                                                       then_node_result.get_element_type(),
                                                       else_node_result.get_element_type()),
                                  "type of then_body output ",
                                  then_node_result.get_element_type(),
                                  " is not equal type of else_body output",
                                  else_node_result.get_element_type());

            // shape inference for output and associated with it body outputs
            auto partial_shape =
                resolve_shape(then_node_result.get_partial_shape(), else_node_result.get_partial_shape());
            set_output_type(output_index, merged_type, partial_shape);
        }
    }
}

std::shared_ptr<ov::Node> ov::op::v8::If::clone_with_new_inputs(const OutputVector& new_args) const {
    OV_OP_SCOPE(v8_If_clone_with_new_inputs);

    check_new_args_count(this, new_args);
    auto op = std::make_shared<op::v8::If>();
    OPENVINO_ASSERT(op.get(),
                    op != nullptr,
                    "Cannot clone ",
                    description(),
                    " operation with name ",
                    get_friendly_name());

    op->set_arguments(new_args);
    op->set_output_size(m_output_descriptions[0].size());
    op->set_then_body(get_then_body()->clone());
    op->set_else_body(get_else_body()->clone());

    for (auto body_index = 0; body_index < 2; ++body_index) {
        for (const auto& m_input_descr : m_input_descriptions[body_index]) {
            op->m_input_descriptions[body_index].push_back(m_input_descr->copy());
        }
        for (const auto& m_output_descr : m_output_descriptions[body_index]) {
            op->m_output_descriptions[body_index].push_back(m_output_descr->copy());
        }
    }
    op->validate_and_infer_types();

    return op;
}

void ov::op::v8::If::set_input(const Output<Node>& value,
                               const std::shared_ptr<v0::Parameter>& then_parameter,
                               const std::shared_ptr<v0::Parameter>& else_parameter) {
    OPENVINO_ASSERT(then_parameter != nullptr || else_parameter != nullptr,
                    "Missing parameters! Both parameters are nullptr!");
    auto then_param_index = m_bodies[THEN_BODY_INDEX]->get_parameter_index(then_parameter);
    auto else_param_index = m_bodies[ELSE_BODY_INDEX]->get_parameter_index(else_parameter);
    OPENVINO_ASSERT(then_parameter == nullptr || then_param_index != -1,
                    "Missing parameter ",
                    then_parameter->get_friendly_name(),
                    " for \'then_body\'!");
    OPENVINO_ASSERT(else_parameter == nullptr || else_param_index != -1,
                    "Missing parameter ",
                    else_parameter->get_friendly_name(),
                    " for \'else_body\'!");
    set_invariant_inputs(value, {then_parameter, else_parameter});
}

ov::Output<ov::Node> ov::op::v8::If::set_output(const std::shared_ptr<v0::Result>& then_result,
                                                const std::shared_ptr<v0::Result>& else_result) {
    OPENVINO_ASSERT(then_result != nullptr, "Incorrect result in \"then_body\"! Result cant be \'nullptr\'");
    OPENVINO_ASSERT(else_result != nullptr, "Incorrect result in \"else_body\"! Result cant be \'nullptr\'");
    auto then_result_id = m_bodies[THEN_BODY_INDEX]->get_result_index(then_result);
    auto else_result_id = m_bodies[ELSE_BODY_INDEX]->get_result_index(else_result);

    OPENVINO_ASSERT(then_result_id != -1, "Missing result ", then_result->get_friendly_name(), "in \'then_body\'!");
    OPENVINO_ASSERT(else_result_id != -1, "Missing result ", else_result->get_friendly_name(), "in \'then_body\'!");

    return set_body_outputs({then_result, else_result});
}
