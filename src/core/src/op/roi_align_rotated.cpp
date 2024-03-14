// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

//#include "openvino/reference/roi_align_rotated.hpp"

#include "itt.hpp"
#include "openvino/op/roi_align_rotated.hpp"
#include "roi_align_rotated_shape_inference.hpp"

using namespace std;

namespace ov {
op::v14::ROIAlignRotated::ROIAlignRotated(const Output<Node>& input,
                           const Output<Node>& rois,
                           const Output<Node>& batch_indices,
                           const int pooled_h,
                           const int pooled_w,
                           const int sampling_ratio,
                           const float spatial_scale,
                           const bool clockwise_mode)
    : Op{{input, rois, batch_indices}},
      m_pooled_h{pooled_h},
      m_pooled_w{pooled_w},
      m_sampling_ratio{sampling_ratio},
      m_spatial_scale{spatial_scale},
      m_clockwise_mode{clockwise_mode} {
    constructor_validate_and_infer_types();
}

void op::v14::ROIAlignRotated::validate_and_infer_types() {
    OV_OP_SCOPE(v14_ROIAlignRotated_validate_and_infer_types);

    const auto out_et = roi_align::validate::data_and_roi_et(this);
    roi_align::validate::batch_indicies_et(this);

    const auto input_shapes = ov::util::get_node_input_partial_shapes(*this);

    auto output_shape = shape_infer(this, input_shapes).front();
    set_output_type(0, out_et, output_shape);

    const auto& input_ps = input_shapes.front();

    // if the channels dimension is not known
    // the first input should be used during the function specialization
    if (input_ps.rank().is_static() && input_ps[1].is_dynamic()) {
        set_input_is_relevant_to_shape(0);
    }
    // if the 'NUM_ROIS' value is not known
    // the last 2 inputs should be used during the function specialization
    if (output_shape[0].is_dynamic()) {
        set_input_is_relevant_to_shape(1);
        set_input_is_relevant_to_shape(2);
    }
}

bool op::v14::ROIAlignRotated::visit_attributes(AttributeVisitor& visitor) {
    OV_OP_SCOPE(v14_ROIAlignRotated_visit_attributes);
    visitor.on_attribute("pooled_h", m_pooled_h);
    visitor.on_attribute("pooled_w", m_pooled_w);
    visitor.on_attribute("sampling_ratio", m_sampling_ratio);
    visitor.on_attribute("spatial_scale", m_spatial_scale);
    visitor.on_attribute("clockwise_mode", m_clockwise_mode);

    return true;
}

shared_ptr<Node> op::v14::ROIAlignRotated::clone_with_new_inputs(const OutputVector& new_args) const {
    OV_OP_SCOPE(v14_ROIAlignRotated_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return make_shared<ROIAlignRotated>(new_args.at(0),
                                 new_args.at(1),
                                 new_args.at(2),
                                 m_pooled_h,
                                 m_pooled_w,
                                 m_sampling_ratio,
                                 m_spatial_scale,
                                 m_clockwise_mode);
}
}
