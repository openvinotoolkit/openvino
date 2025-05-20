// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ov_ops/moe_expert.hpp"

#include "itt.hpp"

namespace ov {
namespace op {
namespace internal {

MOEExpert::MOEExpert(const OutputVector& args, const Attributes& attrs) : Op(args), m_attrs(attrs) {
    constructor_validate_and_infer_types();
}

const MOEExpert::Config& MOEExpert::get_config() const {
    return m_attrs.config;
}

void MOEExpert::set_config(const Config& config) {
    m_attrs.config = config;
}

std::shared_ptr<ov::Node> MOEExpert::clone_with_new_inputs(const ov::OutputVector& new_args) const {
    INTERNAL_OP_SCOPE(internal_MOEExpert_clone_with_new_inputs);
    check_new_args_count(this, new_args);

    return std::make_shared<MOEExpert>(new_args, m_attrs);
}

void MOEExpert::validate_and_infer_types() {
    INTERNAL_OP_SCOPE(internal_MOEExpert_validate_and_infer_types);
    OPENVINO_ASSERT(get_input_size() == 2 || get_input_size() == 4,
                    "MOEExpert must have 2/4 inputs whereas it has ",
                    get_input_size());

    set_output_type(0, get_input_element_type(0), get_input_partial_shape(0));
}

bool MOEExpert::visit_attributes(ov::AttributeVisitor& visitor) {
    INTERNAL_OP_SCOPE(internal_MOEExpert_visit_attributes);
    visitor.start_structure("config");

    visitor.on_attribute("topk", m_attrs.config.topk);
    visitor.on_attribute("expert_num", m_attrs.config.expert_num);
    visitor.on_attribute("hidden_size", m_attrs.config.hidden_size);
    visitor.on_attribute("intermediate_size", m_attrs.config.intermediate_size);
    visitor.on_attribute("group_size", m_attrs.config.group_size);
    visitor.on_attribute("fused_router_logic", m_attrs.config.fused_router_logic);
    visitor.on_attribute("weight_type", m_attrs.config.weight_type);
    visitor.on_attribute("scale_type", m_attrs.config.scale_type);
    visitor.on_attribute("zp_type", m_attrs.config.zp_type);
    visitor.finish_structure();
    m_attrs.consts.resize(m_attrs.config.expert_num);
    for (size_t i = 0; i < m_attrs.config.expert_num; i++) {
        for (size_t j = 0; j < 3; j++) {
            m_attrs.consts[i].gates[j]->visit_attributes(visitor);
        }
    }
    return true;
}

}  // namespace internal
}  // namespace op
}  // namespace ov
