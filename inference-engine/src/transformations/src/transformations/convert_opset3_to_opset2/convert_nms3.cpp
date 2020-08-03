// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/convert_opset3_to_opset2/convert_nms3.hpp"

#include <memory>
#include <vector>

#include <ngraph/opsets/opset1.hpp>
#include <ngraph/opsets/opset2.hpp>
#include <ngraph/opsets/opset3.hpp>
#include <ngraph/rt_info.hpp>

void ngraph::pass::ConvertNMS1ToNMS3::convert_nms1_to_nms3() {
    auto boxes = std::make_shared<pattern::op::Label>(element::f32, Shape{1, 1000, 4});
    auto scores = std::make_shared<pattern::op::Label>(element::f32, Shape{1, 1, 1000});
    auto max_output_boxes_per_class = ngraph::opset3::Constant::create(element::i64, Shape{}, {10});
    auto iou_threshold = ngraph::opset3::Constant::create(element::f32, Shape{}, {0.75});
    auto score_threshold = ngraph::opset3::Constant::create(element::f32, Shape{}, {0.7});
    auto nms = std::make_shared<ngraph::opset1::NonMaxSuppression>(boxes, scores, max_output_boxes_per_class,
                                                                   iou_threshold, score_threshold);

    ngraph::graph_rewrite_callback callback = [](pattern::Matcher &m) {
        auto nms = std::dynamic_pointer_cast<ngraph::opset1::NonMaxSuppression>(m.get_match_root());
        if (!nms) {
            return false;
        }

        auto nms3 = std::make_shared<ngraph::opset3::NonMaxSuppression>(nms->input_value(0), nms->input_value(1),
                nms->input_value(2), nms->input_value(3), nms->input_value(4),
                static_cast<const op::v3::NonMaxSuppression::BoxEncodingType>(nms->get_box_encoding()),
                nms->get_sort_result_descending());

        nms3->set_friendly_name(nms->get_friendly_name());
        ngraph::copy_runtime_info(nms, nms3);
        ngraph::replace_node(nms, nms3);
        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(nms, "ConvertNMS1ToNMS3");
    this->add_matcher(m, callback, PassProperty::CHANGE_DYNAMIC_STATE);
}
