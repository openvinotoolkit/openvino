// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/roi_align_rotated.hpp"

#include "itt.hpp"

namespace ov {
namespace op {
namespace v15 {
ROIAlignRotated::ROIAlignRotated(const Output<Node>& input,
                                 const Output<Node>& rois,
                                 const Output<Node>& batch_indices,
                                 const int pooled_h,
                                 const int pooled_w,
                                 const int sampling_ratio,
                                 const float spatial_scale,
                                 const bool clockwise_mode)
    : ROIAlignBase{input, rois, batch_indices, pooled_h, pooled_w, sampling_ratio, spatial_scale},
      m_clockwise_mode{clockwise_mode} {
    // NOTE: Cannot be called in base class, since then ROIAlignRotated
    // is not fully constructed.
    constructor_validate_and_infer_types();
}

void ROIAlignRotated::validate_and_infer_types() {
    OV_OP_SCOPE(v14_ROIAlignRotated_validate_and_infer_types);
    ROIAlignBase::validate_and_infer_types();
}

bool ROIAlignRotated::visit_attributes(AttributeVisitor& visitor) {
    OV_OP_SCOPE(v14_ROIAlignRotated_visit_attributes);
    ROIAlignBase::visit_attributes(visitor);
    visitor.on_attribute("clockwise_mode", m_clockwise_mode);

    return true;
}

std::shared_ptr<Node> ROIAlignRotated::clone_with_new_inputs(const OutputVector& new_args) const {
    OV_OP_SCOPE(v14_ROIAlignRotated_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return std::make_shared<ROIAlignRotated>(new_args.at(0),
                                             new_args.at(1),
                                             new_args.at(2),
                                             get_pooled_h(),
                                             get_pooled_w(),
                                             get_sampling_ratio(),
                                             get_spatial_scale(),
                                             get_clockwise_mode());
}
}  // namespace v15
}  // namespace op
}  // namespace ov
