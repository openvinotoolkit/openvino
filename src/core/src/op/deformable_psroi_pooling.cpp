// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/deformable_psroi_pooling.hpp"

#include "deformable_psroi_pooling_shape_inference.hpp"
#include "itt.hpp"

namespace ov {

op::v1::DeformablePSROIPooling::DeformablePSROIPooling(const Output<Node>& input,
                                                       const Output<Node>& coords,
                                                       const Output<Node>& offsets,
                                                       const int64_t output_dim,
                                                       const float spatial_scale,
                                                       const int64_t group_size,
                                                       const std::string mode,
                                                       int64_t spatial_bins_x,
                                                       int64_t spatial_bins_y,
                                                       float trans_std,
                                                       int64_t part_size)
    : Op({input, coords, offsets}),
      m_output_dim(output_dim),
      m_spatial_scale(spatial_scale),
      m_group_size(group_size),
      m_mode(mode),
      m_spatial_bins_x(spatial_bins_x),
      m_spatial_bins_y(spatial_bins_y),
      m_trans_std(trans_std),
      m_part_size(part_size) {
    constructor_validate_and_infer_types();
}

op::v1::DeformablePSROIPooling::DeformablePSROIPooling(const Output<Node>& input,
                                                       const Output<Node>& coords,
                                                       const int64_t output_dim,
                                                       const float spatial_scale,
                                                       const int64_t group_size,
                                                       const std::string mode,
                                                       int64_t spatial_bins_x,
                                                       int64_t spatial_bins_y,
                                                       float trans_std,
                                                       int64_t part_size)
    : Op({input, coords}),
      m_output_dim(output_dim),
      m_spatial_scale(spatial_scale),
      m_group_size(group_size),
      m_mode(mode),
      m_spatial_bins_x(spatial_bins_x),
      m_spatial_bins_y(spatial_bins_y),
      m_trans_std(trans_std),
      m_part_size(part_size) {
    constructor_validate_and_infer_types();
}

bool op::v1::DeformablePSROIPooling::visit_attributes(AttributeVisitor& visitor) {
    OV_OP_SCOPE(v1_DeformablePSROIPooling_visit_attributes);
    visitor.on_attribute("output_dim", m_output_dim);
    visitor.on_attribute("spatial_scale", m_spatial_scale);
    visitor.on_attribute("group_size", m_group_size);
    visitor.on_attribute("mode", m_mode);
    visitor.on_attribute("spatial_bins_x", m_spatial_bins_x);
    visitor.on_attribute("spatial_bins_y", m_spatial_bins_y);
    visitor.on_attribute("trans_std", m_trans_std);
    visitor.on_attribute("part_size", m_part_size);
    return true;
}

void op::v1::DeformablePSROIPooling::validate_and_infer_types() {
    OV_OP_SCOPE(v1_DeformablePSROIPooling_validate_and_infer_types);
    const auto& input_et = get_input_element_type(0);
    const auto input_shapes = ov::util::get_node_input_partial_shapes(*this);
    set_output_type(0, input_et, shape_infer(this, input_shapes)[0]);
}

std::shared_ptr<Node> op::v1::DeformablePSROIPooling::clone_with_new_inputs(const OutputVector& new_args) const {
    OV_OP_SCOPE(v1_DeformablePSROIPooling_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    if (new_args.size() == 3) {
        return std::make_shared<v1::DeformablePSROIPooling>(new_args.at(0),
                                                            new_args.at(1),
                                                            new_args.at(2),
                                                            m_output_dim,
                                                            m_spatial_scale,
                                                            m_group_size,
                                                            m_mode,
                                                            m_spatial_bins_x,
                                                            m_spatial_bins_y,
                                                            m_trans_std,
                                                            m_part_size);
    } else if (new_args.size() == 2) {
        return std::make_shared<v1::DeformablePSROIPooling>(new_args.at(0),
                                                            new_args.at(1),
                                                            m_output_dim,
                                                            m_spatial_scale,
                                                            m_group_size,
                                                            m_mode,
                                                            m_spatial_bins_x,
                                                            m_spatial_bins_y,
                                                            m_trans_std,
                                                            m_part_size);
    } else {
        OPENVINO_THROW("Not supported number of DeformablePSROIPooling args");
    }
}

void op::v1::DeformablePSROIPooling::set_output_dim(int64_t output_dim) {
    m_output_dim = output_dim;
}

void op::v1::DeformablePSROIPooling::set_group_size(int64_t group_size) {
    m_group_size = group_size;
}
}  // namespace ov
