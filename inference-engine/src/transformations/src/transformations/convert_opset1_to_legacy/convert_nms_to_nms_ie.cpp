// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/convert_opset1_to_legacy/convert_nms_to_nms_ie.hpp"

#include <memory>
#include <vector>

#include <ngraph/opsets/opset1.hpp>

#include <ngraph_ops/nms_ie.hpp>

void ngraph::pass::ConvertNMSToNMSIE::convert_nms_to_nms_ie() {
    auto input_0 = std::make_shared<pattern::op::Label>(element::f32, Shape{1, 1, 4});
    auto input_1 = std::make_shared<pattern::op::Label>(element::f32, Shape{1, 1, 1});
    auto max_per_class = std::make_shared<pattern::op::Label>(element::i64, Shape{});
    auto iou_threshold = std::make_shared<pattern::op::Label>(element::f32, Shape{});
    auto score_threshold = std::make_shared<pattern::op::Label>(element::f32, Shape{});
    auto nms = std::make_shared<ngraph::opset1::NonMaxSuppression>(input_0, input_1, max_per_class, iou_threshold,
                                                                   score_threshold);

    ngraph::graph_rewrite_callback callback = [](pattern::Matcher &m) {
        auto nms = std::dynamic_pointer_cast<ngraph::opset1::NonMaxSuppression>(m.get_match_root());
        if (!nms) {
            return false;
        }
        if (nms->input(2).get_shape().size() == 1 && nms->input(3).get_shape().size() == 1 &&
            nms->input(4).get_shape().size() == 1) {
            return false;
        }

        auto new_max_per_class = nms->input(2).get_source_output();
        if (nms->input(2).get_shape().empty()) {
            new_max_per_class = std::make_shared<ngraph::op::Unsqueeze>(
                    nms->input(2).get_source_output().get_node_shared_ptr(),
                    opset1::Constant::create(element::i64, Shape{1}, {0}));
        }
        auto new_iou_threshold = nms->input(3).get_source_output();
        if (nms->input(3).get_shape().empty()) {
            new_iou_threshold = std::make_shared<ngraph::op::Unsqueeze>(
                    nms->input(3).get_source_output().get_node_shared_ptr(),
                    opset1::Constant::create(element::i64, Shape{1}, {0}));
        }
        auto new_score_threshold = nms->input(4).get_source_output();
        if (nms->input(4).get_shape().empty()) {
            new_score_threshold = std::make_shared<ngraph::op::Unsqueeze>(
                    nms->input(4).get_source_output().get_node_shared_ptr(),
                    opset1::Constant::create(element::i64, Shape{1}, {0}));
        }
        int center_point_box = 0;
        switch (nms->get_box_encoding()) {
            case ::ngraph::opset1::NonMaxSuppression::BoxEncodingType::CENTER:
                center_point_box = 1;
                break;
            case ::ngraph::opset1::NonMaxSuppression::BoxEncodingType::CORNER:
                center_point_box = 0;
                break;
            default:
                throw ngraph_error("NonMaxSuppression layer " + nms->get_friendly_name() +
                                   " has unsupported box encoding");
        }
        auto new_nms = std::make_shared<ngraph::op::NonMaxSuppressionIE>(nms->input(0).get_source_output(),
                                                                         nms->input(1).get_source_output(),
                                                                         new_max_per_class,
                                                                         new_iou_threshold,
                                                                         new_score_threshold,
                                                                         nms->output(0).get_shape(),
                                                                         center_point_box,
                                                                         nms->get_sort_result_descending());
        new_nms->set_friendly_name(nms->get_friendly_name());
        ngraph::replace_node(m.get_match_root(), new_nms);
        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(nms, "ConvertNMSToNMSIE");
    this->add_matcher(m, callback, PassProperty::CHANGE_DYNAMIC_STATE);
}