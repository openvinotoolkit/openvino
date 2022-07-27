// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/op_conversions/convert_roi_align_v9_to_v3.hpp"

#include <ngraph/opsets/opset3.hpp>
#include <ngraph/opsets/opset9.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>
#include <ngraph/rt_info.hpp>

#include "itt.hpp"

ngraph::pass::ConvertROIAlign9To3::ConvertROIAlign9To3() {
    MATCHER_SCOPE(ConvertROIAlign9To3);

    auto roi_align_v9 = pattern::wrap_type<ngraph::opset9::ROIAlign>();

    ngraph::matcher_pass_callback callback = [=](pattern::Matcher& m) {
        auto roi_align_v9_node = std::dynamic_pointer_cast<ngraph::opset9::ROIAlign>(m.get_match_root());
        if (!roi_align_v9_node)
            return false;

        ngraph::opset9::ROIAlign::AlignedMode aligned_mode_v9 = roi_align_v9_node->get_aligned_mode();
        if (aligned_mode_v9 != ngraph::opset9::ROIAlign::AlignedMode::ASYMMETRIC)
            return false;

        const int pooled_h = roi_align_v9_node->get_pooled_h();
        const int pooled_w = roi_align_v9_node->get_pooled_w();
        const int sampling_ratio = roi_align_v9_node->get_sampling_ratio();
        const float spatial_scale = roi_align_v9_node->get_spatial_scale();
        ngraph::opset9::ROIAlign::PoolingMode m_mode_v9 = roi_align_v9_node->get_mode();
        ngraph::opset3::ROIAlign::PoolingMode m_mode_v3;
        switch (m_mode_v9) {
        case ngraph::opset9::ROIAlign::PoolingMode::AVG: {
            m_mode_v3 = ngraph::opset3::ROIAlign::PoolingMode::AVG;
            break;
        }
        case ngraph::opset9::ROIAlign::PoolingMode::MAX: {
            m_mode_v3 = ngraph::opset3::ROIAlign::PoolingMode::MAX;
            break;
        }
        default: {
            throw ngraph_error("unsupported PoolingMode ");
        }
        }

        auto roi_align_v3 = std::make_shared<ngraph::opset3::ROIAlign>(roi_align_v9_node->input_value(0),
                                                                       roi_align_v9_node->input_value(1),
                                                                       roi_align_v9_node->input_value(2),
                                                                       pooled_h,
                                                                       pooled_w,
                                                                       sampling_ratio,
                                                                       spatial_scale,
                                                                       m_mode_v3);
        roi_align_v3->set_friendly_name(roi_align_v9_node->get_friendly_name());
        ngraph::copy_runtime_info(roi_align_v9_node, roi_align_v3);
        ngraph::replace_node(roi_align_v9_node, roi_align_v3);

        return true;
    };

    auto m = std::make_shared<pattern::Matcher>(roi_align_v9, matcher_name);
    register_matcher(m, callback);
}
