// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/op/grouped_space_to_depth.hpp"

namespace ov::intel_gpu::op {

GroupedSpaceToDepth::GroupedSpaceToDepth(const ov::Output<Node>& input, size_t factor_t, size_t factor_s, size_t output_channels)
    : Op({input}),
      m_factor_t(factor_t),
      m_factor_s(factor_s),
      m_output_channels(output_channels) {
    constructor_validate_and_infer_types();
}

void GroupedSpaceToDepth::validate_and_infer_types() {
    NODE_VALIDATION_CHECK(this, m_factor_t > 0, "factor_t must be greater than zero");
    NODE_VALIDATION_CHECK(this, m_factor_s > 0, "factor_s must be greater than zero");
    NODE_VALIDATION_CHECK(this, m_output_channels > 0, "output_channels must be greater than zero");

    auto output_shape = get_input_partial_shape(0);
    if (output_shape.rank().is_static()) {
        NODE_VALIDATION_CHECK(this, output_shape.rank().get_length() == 5, "input rank must be 5");

        const auto input_channels = output_shape[1];
        const size_t factor_volume = m_factor_t * m_factor_s * m_factor_s;
        if (input_channels.is_static()) {
            NODE_VALIDATION_CHECK(this,
                                  (input_channels.get_length() * factor_volume) % m_output_channels == 0,
                                  "input channels * factor_t * factor_s^2 must be divisible by output_channels");
        }
        if (output_shape[3].is_static()) {
            NODE_VALIDATION_CHECK(this, output_shape[3].get_length() % m_factor_s == 0, "input height must be divisible by factor_s");
        }
        if (output_shape[4].is_static()) {
            NODE_VALIDATION_CHECK(this, output_shape[4].get_length() % m_factor_s == 0, "input width must be divisible by factor_s");
        }

        output_shape[1] = ov::Dimension(static_cast<int64_t>(m_output_channels));
        output_shape[2] = (output_shape[2] + static_cast<int64_t>(m_factor_t - 1)) / static_cast<int64_t>(m_factor_t);
        output_shape[3] = output_shape[3] / static_cast<int64_t>(m_factor_s);
        output_shape[4] = output_shape[4] / static_cast<int64_t>(m_factor_s);
    }

    set_output_type(0, get_input_element_type(0), output_shape);
}

bool GroupedSpaceToDepth::visit_attributes(ov::AttributeVisitor& visitor) {
    visitor.on_attribute("factor_t", m_factor_t);
    visitor.on_attribute("factor_s", m_factor_s);
    visitor.on_attribute("output_channels", m_output_channels);
    return true;
}

std::shared_ptr<ov::Node> GroupedSpaceToDepth::clone_with_new_inputs(const ov::OutputVector& new_args) const {
    check_new_args_count(this, new_args);
    return std::make_shared<GroupedSpaceToDepth>(new_args.at(0), m_factor_t, m_factor_s, m_output_channels);
}

}  // namespace ov::intel_gpu::op
