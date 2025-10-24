// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/op/moe_fused_compressed.hpp"

namespace ov::intel_gpu::op {

MOEFusedCompressed::MOEFusedCompressed(const OutputVector& args, const Config& config) : Op(args), m_config(config) {
    constructor_validate_and_infer_types();
}

const MOEFusedCompressed::Config& MOEFusedCompressed::get_config() const {
    return m_config;
}

void MOEFusedCompressed::set_config(const Config& config) {
    m_config = config;
}

std::shared_ptr<ov::Node> MOEFusedCompressed::clone_with_new_inputs(const ov::OutputVector& new_args) const {
    check_new_args_count(this, new_args);

    return std::make_shared<MOEFusedCompressed>(new_args, m_config);
}

void MOEFusedCompressed::validate_and_infer_types() {
    auto output_type = m_config.out_type == ov::element::dynamic ? get_input_element_type(0) : m_config.out_type;

    set_output_type(0, output_type, get_input_partial_shape(0));
}

bool MOEFusedCompressed::visit_attributes(ov::AttributeVisitor& visitor) {
    visitor.on_attribute("hidden_size", m_config.hidden_size);
    visitor.on_attribute("inter_size", m_config.inter_size);
    visitor.on_attribute("num_expert", m_config.num_expert);
    visitor.on_attribute("top_k", m_config.top_k);
    visitor.on_attribute("group_size", m_config.group_size);
    visitor.on_attribute("out_type", m_config.out_type);

    return true;
}

}  // namespace ov::intel_gpu::op
