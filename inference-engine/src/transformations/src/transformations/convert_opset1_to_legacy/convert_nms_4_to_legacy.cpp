// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <memory>
#include <vector>

#include <ngraph/graph_util.hpp>
#include <ngraph/opsets/opset1.hpp>
#include <ngraph/opsets/opset4.hpp>
#include <ngraph_ops/nms_ie.hpp>
#include <ngraph/rt_info.hpp>
#include <transformations/utils/utils.hpp>

#include "transformations/convert_opset1_to_legacy/convert_nms_4_to_legacy.hpp"

NGRAPH_RTTI_DEFINITION(ngraph::pass::ConvertNMS4ToLegacyMatcher, "ConvertNMS4ToLegacyMatcher", 0);

ngraph::pass::ConvertNMS4ToLegacyMatcher::ConvertNMS4ToLegacyMatcher() {
    auto boxes = std::make_shared<pattern::op::Label>(element::f32, Shape{1, 1000, 4});
    auto scores = std::make_shared<pattern::op::Label>(element::f32, Shape{1, 1, 1000});
    auto max_output_boxes_per_class = ngraph::opset4::Constant::create(element::i64, Shape{}, {10});
    auto iou_threshold = ngraph::opset4::Constant::create(element::f32, Shape{}, {0.75});
    auto score_threshold = ngraph::opset4::Constant::create(element::f32, Shape{}, {0.7});
    auto nms = std::make_shared<ngraph::opset4::NonMaxSuppression>(boxes, scores, max_output_boxes_per_class,
                                                                   iou_threshold, score_threshold);

    ngraph::matcher_pass_callback callback = [](pattern::Matcher &m) {
        auto nms_4 = std::dynamic_pointer_cast<ngraph::opset4::NonMaxSuppression>(m.get_match_root());
        if (!nms_4) {
            return false;
        }

        const auto new_args = nms_4->input_values();
        const auto& arg2 = new_args.size() > 2 ? new_args.at(2) : ngraph::opset4::Constant::create(element::i32, Shape{}, {0});
        const auto& arg3 = new_args.size() > 3 ? new_args.at(3) : ngraph::opset4::Constant::create(element::f32, Shape{}, {.0f});
        const auto& arg4 = new_args.size() > 4 ? new_args.at(4) : ngraph::opset4::Constant::create(element::f32, Shape{}, {.0f});

        const auto max_output_boxes_per_class_rank = arg2.get_partial_shape().rank();
        const auto iou_threshold_rank = arg3.get_partial_shape().rank();
        const auto score_threshold_rank = arg4.get_partial_shape().rank();

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

        auto new_max_per_class = arg2;
        if (max_output_boxes_per_class_rank.get_length() == 0) {
            // WA: we need to create Constant manually because it requires by NMS shape inference
            //     otherwise we will get dynamic shape until first CF is executed. It can be resolved
            //     if CF will be executed right after transformation and before Validate pass.
            if (auto new_max_per_class_const = std::dynamic_pointer_cast<opset1::Constant>(new_max_per_class.get_node_shared_ptr())) {
                new_max_per_class = opset1::Constant::create(element::i64, Shape{1}, new_max_per_class_const->cast_vector<int64_t>());
            } else {
                new_max_per_class = std::make_shared<ngraph::op::Unsqueeze>(arg2, opset1::Constant::create(element::i64, Shape{1}, {0}));
                new_ops.push_back(new_max_per_class.get_node_shared_ptr());
            }
        }
        auto new_iou_threshold = arg3;
        if (iou_threshold_rank.get_length() == 0) {
            new_iou_threshold = std::make_shared<ngraph::op::Unsqueeze>(arg3, opset1::Constant::create(element::i64, Shape{1}, {0}));
            new_ops.push_back(new_iou_threshold.get_node_shared_ptr());
        }
        auto new_score_threshold = arg4;
        if (score_threshold_rank.get_length() == 0) {
            new_score_threshold = std::make_shared<ngraph::op::Unsqueeze>(arg4, opset1::Constant::create(element::i64, Shape{1}, {0}));
            new_ops.push_back(new_score_threshold.get_node_shared_ptr());
        }

        int center_point_box = 0;
        switch (nms_4->get_box_encoding()) {
            case ::ngraph::opset4::NonMaxSuppression::BoxEncodingType::CENTER:
                center_point_box = 1;
                break;
            case ::ngraph::opset4::NonMaxSuppression::BoxEncodingType::CORNER:
                center_point_box = 0;
                break;
            default:
                throw ngraph_error("NonMaxSuppression layer " + nms_4->get_friendly_name() +
                                   " has unsupported box encoding");
        }
        const auto nms_legacy = std::make_shared<op::NonMaxSuppressionIE2>(
                new_args.at(0),
                new_args.at(1),
                new_max_per_class,
                new_iou_threshold,
                new_score_threshold,
                center_point_box,
                nms_4->get_sort_result_descending(),
                nms_4->get_output_type());
        new_ops.push_back(nms_legacy);

        nms_legacy->set_friendly_name(nms_4->get_friendly_name());
        ngraph::copy_runtime_info(nms_4, new_ops);
        ngraph::replace_node(nms_4, nms_legacy);
        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(nms, "ConvertNMS4ToNMSLegacy");
    this->register_matcher(m, callback);
}
