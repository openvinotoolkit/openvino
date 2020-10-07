// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/convert_previous_nms_to_nms_5.hpp"

#include <list>
#include <memory>
#include <vector>

#include <ngraph/opsets/opset1.hpp>
#include <ngraph/opsets/opset3.hpp>
#include <ngraph/opsets/opset4.hpp>
#include <ngraph/opsets/opset5.hpp>
#include <ngraph/rt_info.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>

NGRAPH_RTTI_DEFINITION(ngraph::pass::ConvertNMS4ToNMS5, "ConvertNMS4ToNMS5", 0);

ngraph::pass::ConvertNMS4ToNMS5::ConvertNMS4ToNMS5() {
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

        size_t num_of_args = new_args.size();

        const auto& arg2 = num_of_args > 2 ? new_args.at(2) : ngraph::opset5::Constant::create(element::i32, Shape{}, {0});
        const auto& arg3 = num_of_args > 3 ? new_args.at(3) : ngraph::opset5::Constant::create(element::f32, Shape{}, {.0f});
        const auto& arg4 = num_of_args > 4 ? new_args.at(4) : ngraph::opset5::Constant::create(element::f32, Shape{}, {.0f});
        const auto& arg5 = ngraph::opset5::Constant::create(element::f32, Shape{}, {.0f});

        auto box_encoding = ::ngraph::opset5::NonMaxSuppression::BoxEncodingType::CENTER;
        switch (nms_4->get_box_encoding()) {
            case ::ngraph::opset4::NonMaxSuppression::BoxEncodingType::CENTER:
                box_encoding = ::ngraph::opset5::NonMaxSuppression::BoxEncodingType::CENTER;;
                break;
            case ::ngraph::opset4::NonMaxSuppression::BoxEncodingType::CORNER:
                box_encoding = ::ngraph::opset5::NonMaxSuppression::BoxEncodingType::CORNER;
                break;
            default:
                throw ngraph_error("NonMaxSuppression layer " + nms_4->get_friendly_name() +
                                   " has unsupported box encoding");
        }

        // list of new nGraph operations
        std::list<std::shared_ptr<::ngraph::Node>> new_ops_list;

        new_ops_list.push_front(arg5);
        if (num_of_args <= 4) {
            new_ops_list.push_front(arg4.get_node_shared_ptr());
        }
        if (num_of_args <= 3) {
            new_ops_list.push_front(arg3.get_node_shared_ptr());
        }
        if (num_of_args <= 2) {
            new_ops_list.push_front(arg2.get_node_shared_ptr());
        }

        const auto nms_5 = std::make_shared<ngraph::op::v5::NonMaxSuppression>(
                new_args.at(0),
                new_args.at(1),
                arg2,
                arg3,
                arg4,
                arg5,
                box_encoding,
                nms_4->get_sort_result_descending(),
                nms_4->get_output_type());

        new_ops_list.push_back(nms_5);

        // vector of new nGraph operations
        NodeVector new_ops(new_ops_list.begin(), new_ops_list.end());

        nms_5->set_friendly_name(nms_4->get_friendly_name());
        ngraph::copy_runtime_info(nms_4, new_ops);
        ngraph::replace_node(nms_4, nms_5);
        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(nms, "ConvertNMS4ToNMS5");
    this->register_matcher(m, callback);
}


NGRAPH_RTTI_DEFINITION(ngraph::pass::ConvertNMS3ToNMS5, "ConvertNMS3ToNMS5", 0);

ngraph::pass::ConvertNMS3ToNMS5::ConvertNMS3ToNMS5() {
    auto boxes = std::make_shared<pattern::op::Label>(element::f32, Shape{1, 1000, 4});
    auto scores = std::make_shared<pattern::op::Label>(element::f32, Shape{1, 1, 1000});
    auto max_output_boxes_per_class = ngraph::opset3::Constant::create(element::i64, Shape{}, {10});
    auto iou_threshold = ngraph::opset3::Constant::create(element::f32, Shape{}, {0.75});
    auto score_threshold = ngraph::opset3::Constant::create(element::f32, Shape{}, {0.7});
    auto nms = std::make_shared<ngraph::opset3::NonMaxSuppression>(boxes, scores, max_output_boxes_per_class,
                                                                   iou_threshold, score_threshold);

    ngraph::matcher_pass_callback callback = [](pattern::Matcher &m) {
        auto nms_3 = std::dynamic_pointer_cast<ngraph::opset3::NonMaxSuppression>(m.get_match_root());
        if (!nms_3) {
            return false;
        }

        const auto new_args = nms_3->input_values();

        size_t num_of_args = new_args.size();

        const auto& arg2 = num_of_args > 2 ? new_args.at(2) : ngraph::opset5::Constant::create(element::i32, Shape{}, {0});
        const auto& arg3 = num_of_args > 3 ? new_args.at(3) : ngraph::opset5::Constant::create(element::f32, Shape{}, {.0f});
        const auto& arg4 = num_of_args > 4 ? new_args.at(4) : ngraph::opset5::Constant::create(element::f32, Shape{}, {.0f});
        const auto& arg5 = ngraph::opset5::Constant::create(element::f32, Shape{}, {.0f});

        auto box_encoding = ::ngraph::opset5::NonMaxSuppression::BoxEncodingType::CENTER;
        switch (nms_3->get_box_encoding()) {
            case ::ngraph::opset3::NonMaxSuppression::BoxEncodingType::CENTER:
                center_point_box = ::ngraph::opset5::NonMaxSuppression::BoxEncodingType::CENTER;;
                break;
            case ::ngraph::opset3::NonMaxSuppression::BoxEncodingType::CORNER:
                center_point_box = ::ngraph::opset5::NonMaxSuppression::BoxEncodingType::CORNER;
                break;
            default:
                throw ngraph_error("NonMaxSuppression layer " + nms_3->get_friendly_name() +
                                   " has unsupported box encoding");
        }

        // list of new nGraph operations
        std::list<std::shared_ptr<::ngraph::Node>> new_ops_list;

        new_ops_list.push_front(arg5);
        if (num_of_args <= 4) {
            new_ops_list.push_front(arg4.get_node_shared_ptr());
        }
        if (num_of_args <= 3) {
            new_ops_list.push_front(arg3.get_node_shared_ptr());
        }
        if (num_of_args <= 2) {
            new_ops_list.push_front(arg2.get_node_shared_ptr());
        }

        const auto nms_5 = std::make_shared<ngraph::op::v5::NonMaxSuppression>(
                new_args.at(0),
                new_args.at(1),
                arg2,
                arg3,
                arg4,
                arg5,
                box_encoding,
                nms_3->get_sort_result_descending(),
                nms_3->get_output_type());

        new_ops_list.push_back(nms_5);

        // vector of new nGraph operations
        NodeVector new_ops(new_ops_list.begin(), new_ops_list.end());

        nms_5->set_friendly_name(nms_3->get_friendly_name());
        ngraph::copy_runtime_info(nms_3, new_ops);
        ngraph::replace_node(nms_3, nms_5);
        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(nms, "ConvertNMS3ToNMS5");
    this->register_matcher(m, callback);
}
