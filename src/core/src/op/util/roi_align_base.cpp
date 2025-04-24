// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/util/roi_align_base.hpp"

#include "itt.hpp"
#include "roi_align_shape_utils.hpp"

namespace ov {
namespace op {
namespace util {

ROIAlignBase::ROIAlignBase(const Output<Node>& input,
                           const Output<Node>& rois,
                           const Output<Node>& batch_indices,
                           const int pooled_h,
                           const int pooled_w,
                           const int sampling_ratio,
                           const float spatial_scale)
    : Op{{input, rois, batch_indices}},
      m_pooled_h{pooled_h},
      m_pooled_w{pooled_w},
      m_sampling_ratio{sampling_ratio},
      m_spatial_scale{spatial_scale} {}

void ROIAlignBase::validate_and_infer_types() {
    OV_OP_SCOPE(util_RoiAlignBase_validate_and_infer_types);

    const auto out_et = roi_align::validate::data_and_roi_et(this);
    roi_align::validate::batch_indicies_et(this);

    const auto input_shapes = ov::util::get_node_input_partial_shapes(*this);

    auto output_shape = roi_align::shape_infer(this, input_shapes).front();
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

bool ROIAlignBase::visit_attributes(AttributeVisitor& visitor) {
    OV_OP_SCOPE(util_RoiAlignBase_visit_attributes);
    visitor.on_attribute("pooled_h", m_pooled_h);
    visitor.on_attribute("pooled_w", m_pooled_w);
    visitor.on_attribute("sampling_ratio", m_sampling_ratio);
    visitor.on_attribute("spatial_scale", m_spatial_scale);

    return true;
}

}  // namespace util
}  // namespace op
}  // namespace ov