// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/convert_opset1_to_legacy/convert_nms_to_nms_ie.hpp"

#include <memory>
#include <vector>

#include <ngraph/opsets/opset1.hpp>

#include <ngraph_ops/nms_ie.hpp>
#include <ngraph/rt_info.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>

ngraph::pass::ConvertNMSToNMSIEMatcher::ConvertNMSToNMSIEMatcher() {
    auto nms = ngraph::pattern::wrap_type<opset1::NonMaxSuppression>();

    ngraph::graph_rewrite_callback callback = [](pattern::Matcher &m) {
        auto nms = std::dynamic_pointer_cast<opset1::NonMaxSuppression>(m.get_match_root());
        if (!nms) {
            return false;
        }

        const auto max_output_boxes_per_class_rank = nms->input(2).get_partial_shape().rank();
        const auto iou_threshold_rank = nms->input(3).get_partial_shape().rank();
        const auto score_threshold_rank = nms->input(4).get_partial_shape().rank();
        // Check that required ranks are not dynamic
        if (max_output_boxes_per_class_rank.is_dynamic() ||
            iou_threshold_rank.is_dynamic() ||
            score_threshold_rank.is_dynamic()) {
            return false;
        }

        if (max_output_boxes_per_class_rank.get_length() == 1 &&
            iou_threshold_rank.get_length() == 1 &&
            score_threshold_rank.get_length() == 1) {
            return false;
        }

        // vector of new nGraph operations
        NodeVector new_ops;

        auto new_max_per_class = nms->input_value(2);
        if (max_output_boxes_per_class_rank.get_length() == 0) {
            // WA: we need to create Constant manually because it requires by NMS shape inference
            //     otherwise we will get dynamic shape until first CF is executed. It can be resolved
            //     if CF will be executed right after transformation and before Validate pass.
            if (auto new_max_per_class_const = std::dynamic_pointer_cast<opset1::Constant>(new_max_per_class.get_node_shared_ptr())) {
                new_max_per_class = opset1::Constant::create(element::i64, Shape{1}, new_max_per_class_const->cast_vector<int64_t>());
            } else {
                new_max_per_class = std::make_shared<ngraph::op::Unsqueeze>(
                        nms->input_value(2),
                        opset1::Constant::create(element::i64, Shape{1}, {0}));
                new_ops.push_back(new_max_per_class.get_node_shared_ptr());
            }
        }
        auto new_iou_threshold = nms->input_value(3);
        if (iou_threshold_rank.get_length() == 0) {
            new_iou_threshold = std::make_shared<ngraph::op::Unsqueeze>(
                    nms->input_value(3),
                    opset1::Constant::create(element::i64, Shape{1}, {0}));
            new_ops.push_back(new_iou_threshold.get_node_shared_ptr());
        }
        auto new_score_threshold = nms->input_value(4);
        if (score_threshold_rank.get_length() == 0) {
            new_score_threshold = std::make_shared<ngraph::op::Unsqueeze>(
                    nms->input_value(4),
                    opset1::Constant::create(element::i64, Shape{1}, {0}));
            new_ops.push_back(new_score_threshold.get_node_shared_ptr());
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
        auto new_nms = std::make_shared<ngraph::op::NonMaxSuppressionIE>(nms->input_value(0),
                                                                         nms->input_value(1),
                                                                         new_max_per_class,
                                                                         new_iou_threshold,
                                                                         new_score_threshold,
                                                                         center_point_box,
                                                                         nms->get_sort_result_descending());
        new_ops.push_back(new_nms);
        new_nms->set_friendly_name(nms->get_friendly_name());
        ngraph::copy_runtime_info(nms, new_ops);
        ngraph::replace_node(nms, new_nms);
        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(nms, "ConvertNMSToNMSIE");
    this->register_matcher(m, callback);
}