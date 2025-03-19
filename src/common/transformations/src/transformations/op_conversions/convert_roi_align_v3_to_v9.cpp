// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/op_conversions/convert_roi_align_v3_to_v9.hpp"

#include "itt.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/op/roi_align.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"

ov::pass::ConvertROIAlign3To9::ConvertROIAlign3To9() {
    MATCHER_SCOPE(ConvertROIAlign3To9);

    auto roi_align_v3 = pattern::wrap_type<ov::op::v3::ROIAlign>();

    matcher_pass_callback callback = [=](pattern::Matcher& m) {
        auto roi_align_v3_node = ov::as_type_ptr<ov::op::v3::ROIAlign>(m.get_match_root());
        if (!roi_align_v3_node)
            return false;

        const int pooled_h = roi_align_v3_node->get_pooled_h();
        const int pooled_w = roi_align_v3_node->get_pooled_w();
        const int sampling_ratio = roi_align_v3_node->get_sampling_ratio();
        const float spatial_scale = roi_align_v3_node->get_spatial_scale();
        ov::op::v3::ROIAlign::PoolingMode m_mode_v3 = roi_align_v3_node->get_mode();
        ov::op::v9::ROIAlign::PoolingMode m_mode_v9;
        switch (m_mode_v3) {
        case ov::op::v3::ROIAlign::PoolingMode::AVG: {
            m_mode_v9 = ov::op::v9::ROIAlign::PoolingMode::AVG;
            break;
        }
        case ov::op::v3::ROIAlign::PoolingMode::MAX: {
            m_mode_v9 = ov::op::v9::ROIAlign::PoolingMode::MAX;
            break;
        }
        default: {
            OPENVINO_THROW("unsupported PoolingMode ");
        }
        }

        auto roi_align_v9 = std::make_shared<ov::op::v9::ROIAlign>(roi_align_v3_node->input_value(0),
                                                                   roi_align_v3_node->input_value(1),
                                                                   roi_align_v3_node->input_value(2),
                                                                   pooled_h,
                                                                   pooled_w,
                                                                   sampling_ratio,
                                                                   spatial_scale,
                                                                   m_mode_v9);
        roi_align_v9->set_friendly_name(roi_align_v3_node->get_friendly_name());
        ov::copy_runtime_info(roi_align_v3_node, roi_align_v9);
        ov::replace_node(roi_align_v3_node, roi_align_v9);

        return true;
    };

    auto m = std::make_shared<pattern::Matcher>(roi_align_v3, matcher_name);
    register_matcher(m, callback);
}
