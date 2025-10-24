// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/op/moe_compressed.hpp"

namespace ov::intel_gpu::op {
MOECompressed::MOECompressed(const OutputVector& args, const Config& config) : MOE(args), m_config(config) {
    constructor_validate_and_infer_types();
}

const MOECompressed::Config& MOECompressed::get_config() const {
    return m_config;
}

void MOECompressed::set_config(const Config& config) {
    m_config = config;
}

std::shared_ptr<ov::Node> MOECompressed::clone_with_new_inputs(const ov::OutputVector& new_args) const {
    check_new_args_count(this, new_args);

    return std::make_shared<MOECompressed>(new_args, m_config);
}

void MOECompressed::validate_and_infer_types() {
    auto output_type = m_config.out_type == ov::element::dynamic ? get_input_element_type(0) : m_config.out_type;

    set_output_type(0, output_type, get_input_partial_shape(0));
}

bool MOECompressed::visit_attributes(ov::AttributeVisitor& visitor) {
    visitor.on_attribute("out_type", m_config.out_type);
    visitor.on_attribute("group_size", m_config.group_size);

    return true;
}

}  // namespace ov::intel_gpu::op
