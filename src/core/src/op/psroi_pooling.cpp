// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/psroi_pooling.hpp"

#include "itt.hpp"
#include "openvino/core/attribute_visitor.hpp"
#include "psroi_pooling_shape_inference.hpp"

namespace ov {
namespace op {
namespace v0 {

PSROIPooling::PSROIPooling(const Output<Node>& input,
                           const Output<Node>& coords,
                           const size_t output_dim,
                           const size_t group_size,
                           const float spatial_scale,
                           int spatial_bins_x,
                           int spatial_bins_y,
                           const std::string& mode)
    : Op({input, coords}),
      m_output_dim(output_dim),
      m_group_size(group_size),
      m_spatial_scale(spatial_scale),
      m_spatial_bins_x(spatial_bins_x),
      m_spatial_bins_y(spatial_bins_y),
      m_mode(mode) {
    constructor_validate_and_infer_types();
}

bool PSROIPooling::visit_attributes(AttributeVisitor& visitor) {
    OV_OP_SCOPE(v0_PSROIPooling_visit_attributes);
    visitor.on_attribute("output_dim", m_output_dim);
    visitor.on_attribute("group_size", m_group_size);
    visitor.on_attribute("spatial_scale", m_spatial_scale);
    visitor.on_attribute("mode", m_mode);
    visitor.on_attribute("spatial_bins_x", m_spatial_bins_x);
    visitor.on_attribute("spatial_bins_y", m_spatial_bins_y);
    return true;
}

void PSROIPooling::validate_and_infer_types() {
    OV_OP_SCOPE(v0_PSROIPooling_validate_and_infer_types);
    const auto& feat_maps_et = get_input_element_type(0);
    const auto& coords_et = get_input_element_type(1);
    NODE_VALIDATION_CHECK(this,
                          feat_maps_et.is_real(),
                          "Feature maps' data type must be floating point. Got " + feat_maps_et.to_string());
    NODE_VALIDATION_CHECK(this,
                          coords_et.is_real(),
                          "Coords' data type must be floating point. Got " + coords_et.to_string());

    const auto output_shapes = shape_infer(this, ov::util::get_node_input_partial_shapes(*this));
    set_output_type(0, feat_maps_et, output_shapes[0]);
}

std::shared_ptr<Node> PSROIPooling::clone_with_new_inputs(const OutputVector& new_args) const {
    OV_OP_SCOPE(v0_PSROIPooling_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return std::make_shared<PSROIPooling>(new_args.at(0),
                                          new_args.at(1),
                                          m_output_dim,
                                          m_group_size,
                                          m_spatial_scale,
                                          m_spatial_bins_x,
                                          m_spatial_bins_y,
                                          m_mode);
}

void PSROIPooling::set_output_dim(size_t output_dim) {
    m_output_dim = output_dim;
}

void PSROIPooling::set_group_size(size_t group_size) {
    m_group_size = group_size;
}

void PSROIPooling::set_spatial_scale(float scale) {
    m_spatial_scale = scale;
}

void PSROIPooling::set_spatial_bins_x(int x) {
    m_spatial_bins_x = x;
}

void PSROIPooling::set_spatial_bins_y(int y) {
    m_spatial_bins_y = y;
}

void PSROIPooling::set_mode(std::string mode) {
    m_mode = std::move(mode);
}
}  // namespace v0
}  // namespace op
}  // namespace ov
