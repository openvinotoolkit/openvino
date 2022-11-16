// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/op/psroi_pooling.hpp"

#include "itt.hpp"
#include "ngraph/attribute_visitor.hpp"

using namespace std;
using namespace ngraph;

BWDCMP_RTTI_DEFINITION(ov::op::v0::PSROIPooling);

ov::op::v0::PSROIPooling::PSROIPooling(const Output<Node>& input,
                                       const Output<Node>& coords,
                                       const size_t output_dim,
                                       const size_t group_size,
                                       const float spatial_scale,
                                       int spatial_bins_x,
                                       int spatial_bins_y,
                                       const string& mode)
    : Op({input, coords}),
      m_output_dim(output_dim),
      m_group_size(group_size),
      m_spatial_scale(spatial_scale),
      m_spatial_bins_x(spatial_bins_x),
      m_spatial_bins_y(spatial_bins_y),
      m_mode(mode) {
    constructor_validate_and_infer_types();
}

bool ngraph::op::v0::PSROIPooling::visit_attributes(AttributeVisitor& visitor) {
    OV_OP_SCOPE(v0_PSROIPooling_visit_attributes);
    visitor.on_attribute("output_dim", m_output_dim);
    visitor.on_attribute("group_size", m_group_size);
    visitor.on_attribute("spatial_scale", m_spatial_scale);
    visitor.on_attribute("mode", m_mode);
    visitor.on_attribute("spatial_bins_x", m_spatial_bins_x);
    visitor.on_attribute("spatial_bins_y", m_spatial_bins_y);
    return true;
}

void ov::op::v0::PSROIPooling::validate_and_infer_types() {
    OV_OP_SCOPE(v0_PSROIPooling_validate_and_infer_types);
    auto feat_maps_et = get_input_element_type(0);
    auto coords_et = get_input_element_type(1);
    NODE_VALIDATION_CHECK(this,
                          feat_maps_et.is_real(),
                          "Feature maps' data type must be floating point. Got " + feat_maps_et.get_type_name());
    NODE_VALIDATION_CHECK(this,
                          coords_et.is_real(),
                          "Coords' data type must be floating point. Got " + coords_et.get_type_name());
    NODE_VALIDATION_CHECK(this,
                          m_mode == "average" || m_mode == "bilinear",
                          "Expected 'average' or 'bilinear' mode. Got " + m_mode);
    NODE_VALIDATION_CHECK(this, m_group_size > 0, "group_size has to be greater than 0");
    if (m_mode == "bilinear") {
        NODE_VALIDATION_CHECK(this, m_spatial_bins_x > 0, "spatial_bins_x has to be greater than 0");
        NODE_VALIDATION_CHECK(this, m_spatial_bins_y > 0, "spatial_bins_y has to be greater than 0");
    }

    const ov::PartialShape& feat_map_pshape = get_input_partial_shape(0);
    const ov::PartialShape& coords_pshape = get_input_partial_shape(1);
    if (feat_map_pshape.rank().is_dynamic() || coords_pshape.rank().is_dynamic()) {
        set_output_type(0, feat_maps_et, ov::PartialShape::dynamic());
    } else {
        NODE_VALIDATION_CHECK(this,
                              feat_map_pshape.rank().get_length() == 4,
                              "PSROIPooling expects 4 dimensions for input. Got ",
                              feat_map_pshape.rank().get_length());
        NODE_VALIDATION_CHECK(this,
                              coords_pshape.rank().get_length() == 2,
                              "PSROIPooling expects 2 dimensions for box coordinates. Got ",
                              coords_pshape.rank().get_length());

        if (feat_map_pshape[1].is_static()) {
            auto num_input_channels = feat_map_pshape[1].get_interval().get_min_val();
            if (m_mode == "average") {
                NODE_VALIDATION_CHECK(this,
                                      num_input_channels % (m_group_size * m_group_size) == 0,
                                      "Number of input's channels must be a multiply of group_size * group_size");
                NODE_VALIDATION_CHECK(this,
                                      m_output_dim == num_input_channels / (m_group_size * m_group_size),
                                      "output_dim must be equal to input channels divided by "
                                      "group_size * group_size");
            } else if (m_mode == "bilinear") {
                NODE_VALIDATION_CHECK(this,
                                      num_input_channels % (m_spatial_bins_x * m_spatial_bins_y) == 0,
                                      "Number of input's channels must be a multiply of "
                                      "spatial_bins_x * spatial_bins_y");
                NODE_VALIDATION_CHECK(
                    this,
                    m_output_dim == static_cast<size_t>(num_input_channels / (m_spatial_bins_x * m_spatial_bins_y)),
                    "output_dim must be equal to input channels divided by "
                    "spatial_bins_x * spatial_bins_y");
            }
        }
        std::vector<Dimension> output_shape{coords_pshape[0], static_cast<Dimension::value_type>(m_output_dim)};
        for (int64_t i = 2; i < feat_map_pshape.rank().get_length(); i++) {
            output_shape.emplace_back(m_group_size);
        }
        set_output_type(0, feat_maps_et, output_shape);
    }
}

shared_ptr<Node> ov::op::v0::PSROIPooling::clone_with_new_inputs(const OutputVector& new_args) const {
    OV_OP_SCOPE(v0_PSROIPooling_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return make_shared<PSROIPooling>(new_args.at(0),
                                     new_args.at(1),
                                     m_output_dim,
                                     m_group_size,
                                     m_spatial_scale,
                                     m_spatial_bins_x,
                                     m_spatial_bins_y,
                                     m_mode);
}
