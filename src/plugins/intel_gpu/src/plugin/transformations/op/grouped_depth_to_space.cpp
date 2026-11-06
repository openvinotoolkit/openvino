// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/op/grouped_depth_to_space.hpp"

namespace ov::intel_gpu::op {

GroupedDepthToSpace::GroupedDepthToSpace(const ov::Output<Node>& input, size_t factor_t, size_t factor_s, size_t output_channels, size_t crop_begin_t)
    : Op({input}),
      m_factor_t(factor_t),
      m_factor_s(factor_s),
      m_output_channels(output_channels),
      m_crop_begin_t(crop_begin_t) {
    constructor_validate_and_infer_types();
}

void GroupedDepthToSpace::validate_and_infer_types() {
    NODE_VALIDATION_CHECK(this, m_factor_t > 0, "factor_t must be greater than zero");
    NODE_VALIDATION_CHECK(this, m_factor_s > 0, "factor_s must be greater than zero");
    NODE_VALIDATION_CHECK(this, m_output_channels > 0, "output_channels must be greater than zero");
    NODE_VALIDATION_CHECK(this, m_crop_begin_t < m_factor_t, "crop_begin_t must be less than factor_t");

    auto output_shape = get_input_partial_shape(0);
    if (output_shape.rank().is_static()) {
        NODE_VALIDATION_CHECK(this, output_shape.rank().get_length() == 5, "input rank must be 5");

        const auto input_channels = output_shape[1];
        const size_t factor = m_factor_t * m_factor_s * m_factor_s;
        if (input_channels.is_static()) {
            NODE_VALIDATION_CHECK(this, input_channels.get_length() > 0, "input channels must be greater than zero");
            NODE_VALIDATION_CHECK(this,
                                  (m_output_channels * factor) % input_channels.get_length() == 0,
                                  "output_channels * factor_t * factor_s^2 must be divisible by input channels");
        }

        output_shape[1] = ov::Dimension(static_cast<int64_t>(m_output_channels));
        output_shape[2] = output_shape[2] * static_cast<int64_t>(m_factor_t) - static_cast<int64_t>(m_crop_begin_t);
        output_shape[3] = output_shape[3] * static_cast<int64_t>(m_factor_s);
        output_shape[4] = output_shape[4] * static_cast<int64_t>(m_factor_s);
    }

    set_output_type(0, get_input_element_type(0), output_shape);
}

bool GroupedDepthToSpace::visit_attributes(ov::AttributeVisitor& visitor) {
    visitor.on_attribute("factor_t", m_factor_t);
    visitor.on_attribute("factor_s", m_factor_s);
    visitor.on_attribute("output_channels", m_output_channels);
    visitor.on_attribute("crop_begin_t", m_crop_begin_t);
    return true;
}

std::shared_ptr<ov::Node> GroupedDepthToSpace::clone_with_new_inputs(const ov::OutputVector& new_args) const {
    check_new_args_count(this, new_args);
    return std::make_shared<GroupedDepthToSpace>(new_args.at(0), m_factor_t, m_factor_s, m_output_channels, m_crop_begin_t);
}

}  // namespace ov::intel_gpu::op
