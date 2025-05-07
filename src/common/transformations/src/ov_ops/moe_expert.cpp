// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ov_ops/moe_expert.hpp"

#include "itt.hpp"

namespace ov {
namespace op {
namespace internal {

MOEExpert::MOEExpert(const OutputVector& args, const Config& cfg, const std::vector<ConstsPerExpert>& consts) : Op(args), m_config(cfg), m_consts(consts) {
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

    return std::make_shared<MOEExpert>(new_args, m_config, m_consts);
}

void MOEExpert::validate_and_infer_types() {
    INTERNAL_OP_SCOPE(internal_MOEExpert_validate_and_infer_types);
    OPENVINO_ASSERT(get_input_size() == 2 || get_input_size() == 4, "MOEExpert must have 2/4 inputs whereas it has ", get_input_size());
    OPENVINO_ASSERT(get_output_size() == 1, "MOEExpert must have 1 output whereas it has ", get_output_size());

    set_output_type(0, get_input_element_type(0), get_input_partial_shape(0));
}

bool MOEExpert::visit_attributes(ov::AttributeVisitor& visitor) {
    INTERNAL_OP_SCOPE(internal_MOEExpert_visit_attributes);
    visitor.start_structure("config");

    visitor.on_attribute("topk", m_config.topk);
    visitor.on_attribute("expert_num", m_config.expert_num);
    visitor.on_attribute("hidden_size", m_config.hidden_size);
    visitor.on_attribute("fused_router_logic", m_config.fused_router_logic);
    visitor.finish_structure();
    m_consts.resize(m_config.expert_num);
    for (size_t i = 0; i < m_config.expert_num; i++) {
        for (size_t j = 0; j < 3; j++) {
            visitor.start_structure("expert" + std::to_string(i) + "_gate" + std::to_string(j));
            m_consts[i].gate[j]->visit_attributes(visitor);
            visitor.finish_structure();
        }
    }
    return true;
}

}  // namespace internal
}  // namespace op
}  // namespace ov
