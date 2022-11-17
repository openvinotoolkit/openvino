// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/op/roi_pooling.hpp"

#include "itt.hpp"

using namespace std;
using namespace ngraph;

BWDCMP_RTTI_DEFINITION(op::v0::ROIPooling);

op::ROIPooling::ROIPooling(const Output<Node>& input,
                           const Output<Node>& coords,
                           const ov::Shape& output_size,
                           const float spatial_scale,
                           const string& method)
    : Op({input, coords}),
      m_output_size(output_size),
      m_spatial_scale(spatial_scale),
      m_method(method) {
    constructor_validate_and_infer_types();
}

void op::ROIPooling::validate_and_infer_types() {
    OV_OP_SCOPE(v0_ROIPooling_validate_and_infer_types);
    auto feat_maps_et = get_input_element_type(0);
    auto coords_et = get_input_element_type(1);
    NODE_VALIDATION_CHECK(this,
                          feat_maps_et.is_real() && coords_et.is_real(),
                          "The data type for input and ROIs is expected to be a floating point type. Got: ",
                          feat_maps_et,
                          " and: ",
                          coords_et);

    NODE_VALIDATION_CHECK(this,
                          feat_maps_et == coords_et,
                          "Type of feature maps (inputs) and rois is expected to be the same. Got: ",
                          feat_maps_et,
                          " and: ",
                          coords_et);

    NODE_VALIDATION_CHECK(this,
                          m_output_size.size() == 2,
                          "The dimension of pooled size is expected to be equal to 2. Got: ",
                          m_output_size.size());

    NODE_VALIDATION_CHECK(this,
                          m_output_size[0] > 0 && m_output_size[1] > 0,
                          "Pooled size attributes pooled_h and pooled_w should should be "
                          "non-negative integers. Got: ",
                          m_output_size[0],
                          " and: ",
                          m_output_size[1],
                          "respectively");

    NODE_VALIDATION_CHECK(this,
                          m_spatial_scale > 0,
                          "The spatial scale attribute should be a positive floating point number. Got: ",
                          m_spatial_scale);

    NODE_VALIDATION_CHECK(this,
                          m_method == "max" || m_method == "bilinear",
                          "Pooling method attribute should be either \'max\' or \'bilinear\'. Got: ",
                          m_method);

    const auto& feat_maps_ps = get_input_partial_shape(0);
    NODE_VALIDATION_CHECK(this,
                          feat_maps_ps.rank().compatible(4),
                          "Expected a 4D tensor for the feature maps input. Got: ",
                          feat_maps_ps);

    const auto& coords_ps = get_input_partial_shape(1);
    NODE_VALIDATION_CHECK(this,
                          coords_ps.rank().compatible(2),
                          "Expected a 2D tensor for the ROIs input with box coordinates. Got: ",
                          coords_ps);

    if (coords_ps.rank().is_static()) {
        const auto coords_second_dim = coords_ps[1];
        NODE_VALIDATION_CHECK(this,
                              coords_second_dim.compatible(5),
                              "The second dimension of ROIs input should contain batch id and box coordinates. ",
                              "This dimension is expected to be equal to 5. Got: ",
                              coords_second_dim);
    }

    // output shape should be {NUM_ROIS, C, pooled_h, pooled_w}
    auto output_shape = ov::PartialShape{{Dimension::dynamic(),
                                          Dimension::dynamic(),
                                          Dimension{static_cast<int64_t>(m_output_size[0])},
                                          Dimension{static_cast<int64_t>(m_output_size[1])}}};

    if (coords_ps.rank().is_static()) {
        output_shape[0] = coords_ps[0];
    }

    if (feat_maps_ps.rank().is_static()) {
        output_shape[1] = feat_maps_ps[1];
    }

    set_output_size(1);
    set_output_type(0, feat_maps_et, output_shape);

    // if channel dimension, C, not known
    // feature maps input is used by shape specialization pass
    if (feat_maps_ps.rank().is_static() && feat_maps_ps[1].is_dynamic()) {
        set_input_is_relevant_to_shape(0);
    }

    // if number of ROIs, NUM_ROIS, not known
    // coordinate input is used by shape specialization pass
    if (coords_ps.rank().is_static() && coords_ps[0].is_dynamic()) {
        set_input_is_relevant_to_shape(1);
    }
}

shared_ptr<Node> op::ROIPooling::clone_with_new_inputs(const OutputVector& new_args) const {
    OV_OP_SCOPE(v0_ROIPooling_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return make_shared<ROIPooling>(new_args.at(0), new_args.at(1), m_output_size, m_spatial_scale, m_method);
}

bool op::ROIPooling::visit_attributes(AttributeVisitor& visitor) {
    OV_OP_SCOPE(v0_ROIPooling_visit_attributes);
    visitor.on_attribute("output_size", m_output_size);
    visitor.on_attribute("pooled_h", m_output_size[0]);
    visitor.on_attribute("pooled_w", m_output_size[1]);
    visitor.on_attribute("spatial_scale", m_spatial_scale);
    visitor.on_attribute("method", m_method);
    return true;
}
