// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ov_ops/moe_expert.hpp"

#include "itt.hpp"

namespace ov {
namespace op {
namespace internal {

MOEExpert::MOEExpert(const OutputVector& args, const Config& cfg, const std::shared_ptr<ov::Model>& body) : SubGraphOp(args), m_config(cfg) {
    SubGraphOp::set_function(body);
    if (cfg.has_non_zero) {
        for (size_t i = 0; i < body->get_parameters().size(); ++i)
            m_input_descriptions[0].push_back(std::make_shared<InvariantInputDescription>(i, i));
    } else {
        m_input_descriptions[0].push_back(std::make_shared<InvariantInputDescription>(0, 0));
        m_input_descriptions[0].push_back(std::make_shared<InvariantInputDescription>(2, 1));
        m_input_descriptions[0].push_back(std::make_shared<InvariantInputDescription>(3, 2));
    }

    for (size_t i = 0; i < body->get_output_size(); ++i)
        m_output_descriptions[0].push_back(std::make_shared<BodyOutputDescription>(i, i));
    constructor_validate_and_infer_types();
}

const MOEExpert::Config& MOEExpert::get_config() const {
    return m_config;
}

void MOEExpert::set_config(const Config& config) {
    m_config = config;
}

std::shared_ptr<ov::Node> MOEExpert::clone_with_new_inputs(const ov::OutputVector& new_args) const {
    INTERNAL_OP_SCOPE(internal_MOEExpert_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return std::make_shared<MOEExpert>(new_args, m_config, get_function()->clone());
}

void MOEExpert::validate_and_infer_types() {
    INTERNAL_OP_SCOPE(internal_MOEExpert_validate_and_infer_types);
    OPENVINO_ASSERT(get_input_size() == 4, "MOEExpert must have 4 inputs whereas it has ", get_input_size());
    OPENVINO_ASSERT(get_output_size() == 1, "MOEExpert must have 1 output whereas it has ", get_output_size());
    const auto& body = get_function();
    OPENVINO_ASSERT(body, "MOEExpert must have initialized body");
    // with non zero can infer subgraph
    if (m_config.has_non_zero)
        validate_and_infer_type_body(body, m_input_descriptions[0]);

    set_output_type(0, get_input_element_type(0), get_input_partial_shape(0));
}

bool MOEExpert::visit_attributes(ov::AttributeVisitor& visitor) {
    INTERNAL_OP_SCOPE(internal_MOEExpert_visit_attributes);
    visitor.start_structure("config");

    visitor.on_attribute("topk", m_config.topk);
    visitor.on_attribute("expert_num", m_config.expert_num);
    visitor.on_attribute("hidden_size", m_config.hidden_size);
    visitor.on_attribute("expert_no", m_config.expert_no);
    visitor.on_attribute("has_non_zero", m_config.has_non_zero);
    visitor.finish_structure();

    visitor.on_attribute("body", m_bodies[0]);
    visitor.on_attribute("input_descriptions", m_input_descriptions[0]);
    return true;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////
MOEExpert2::MOEExpert2(const OutputVector& args, const Config& cfg, const std::vector<std::shared_ptr<ov::Model>>& body) : Op(args), m_config(cfg), m_body(body) {
    constructor_validate_and_infer_types();
}

const MOEExpert2::Config& MOEExpert2::get_config() const {
    return m_config;
}

void MOEExpert2::set_config(const Config& config) {
    m_config = config;
}

std::shared_ptr<ov::Node> MOEExpert2::clone_with_new_inputs(const ov::OutputVector& new_args) const {
    INTERNAL_OP_SCOPE(internal_MOEExpert2_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    std::vector<std::shared_ptr<ov::Model>> models(m_body.size());
    for (size_t i = 0; i < m_body.size(); i++) {
        models[i] = m_body[i]->clone();
    }
    return std::make_shared<MOEExpert2>(new_args, m_config, models);
}

void MOEExpert2::validate_and_infer_types() {
    INTERNAL_OP_SCOPE(internal_MOEExpert2_validate_and_infer_types);
    OPENVINO_ASSERT(get_input_size() == 4, "MOEExpert2 must have 4 inputs whereas it has ", get_input_size());
    OPENVINO_ASSERT(get_output_size() == 1, "MOEExpert2 must have 1 output whereas it has ", get_output_size());

    set_output_type(0, get_input_element_type(0), get_input_partial_shape(0));
}

bool MOEExpert2::visit_attributes(ov::AttributeVisitor& visitor) {
    INTERNAL_OP_SCOPE(internal_MOEExpert2_visit_attributes);
    visitor.start_structure("config");

    visitor.on_attribute("topk", m_config.topk);
    visitor.on_attribute("expert_num", m_config.expert_num);
    visitor.on_attribute("hidden_size", m_config.hidden_size);
    visitor.on_attribute("expert_no", m_config.expert_no);
    visitor.finish_structure();
    OPENVINO_ASSERT(m_config.expert_num == m_body.size());
    m_body.resize(m_config.expert_num);
    for (size_t i = 0; i < m_config.expert_num; i++)
        visitor.on_attribute("body" + std::to_string(i), m_body[i]);
    return true;
}

}  // namespace internal
}  // namespace op
}  // namespace ov
