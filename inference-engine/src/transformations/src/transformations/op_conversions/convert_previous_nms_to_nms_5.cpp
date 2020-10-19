// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/op_conversions/convert_previous_nms_to_nms_5.hpp"

#include <list>
#include <memory>
#include <vector>

#include <ngraph/opsets/opset1.hpp>
#include <ngraph/opsets/opset3.hpp>
#include <ngraph/opsets/opset4.hpp>
#include <ngraph/opsets/opset5.hpp>
#include <ngraph/rt_info.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>

using NMS5BoxEncoding = ngraph::opset5::NonMaxSuppression::BoxEncodingType;

struct NMSAttributes {
    ngraph::element::Type output_type;
    NMS5BoxEncoding box_encoding;
    bool sort_result_descending;
    bool is_supported_nms;
};

namespace {
    std::shared_ptr<ngraph::opset4::NonMaxSuppression> nms4_pattern() {
        auto boxes = std::make_shared<pattern::op::Label>(element::f32, Shape{1, 1000, 4});
        auto scores = std::make_shared<pattern::op::Label>(element::f32, Shape{1, 1, 1000});
        auto max_output_boxes_per_class = ngraph::opset4::Constant::create(element::i64, Shape{}, {10});
        auto iou_threshold = ngraph::opset4::Constant::create(element::f32, Shape{}, {0.75});
        auto score_threshold = ngraph::opset4::Constant::create(element::f32, Shape{}, {0.7});
        auto nms = std::make_shared<ngraph::opset4::NonMaxSuppression>(boxes, scores, max_output_boxes_per_class,
                                                                       iou_threshold, score_threshold);
        return nms;
    }

    std::shared_ptr<ngraph::opset3::NonMaxSuppression> nms3_pattern() {
        auto boxes = std::make_shared<pattern::op::Label>(element::f32, Shape{1, 1000, 4});
        auto scores = std::make_shared<pattern::op::Label>(element::f32, Shape{1, 1, 1000});
        auto max_output_boxes_per_class = ngraph::opset3::Constant::create(element::i64, Shape{}, {10});
        auto iou_threshold = ngraph::opset3::Constant::create(element::f32, Shape{}, {0.75});
        auto score_threshold = ngraph::opset3::Constant::create(element::f32, Shape{}, {0.7});
        auto nms = std::make_shared<ngraph::opset3::NonMaxSuppression>(boxes, scores, max_output_boxes_per_class,
                                                                       iou_threshold, score_threshold);
        return nms;
    }

    std::shared_ptr<ngraph::opset1::NonMaxSuppression> nms1_pattern() {
        auto boxes = std::make_shared<pattern::op::Label>(element::f32, Shape{1, 1000, 4});
        auto scores = std::make_shared<pattern::op::Label>(element::f32, Shape{1, 1, 1000});
        auto max_output_boxes_per_class = ngraph::opset1::Constant::create(element::i64, Shape{}, {10});
        auto iou_threshold = ngraph::opset1::Constant::create(element::f32, Shape{}, {0.75});
        auto score_threshold = ngraph::opset1::Constant::create(element::f32, Shape{}, {0.7});
        auto nms = std::make_shared<ngraph::opset1::NonMaxSuppression>(boxes, scores, max_output_boxes_per_class,
                                                                       iou_threshold, score_threshold);
        return nms;
    }

    NMSAttributes get_nms4_attrs(const std::shared_ptr<ngraph::opset4::NonMaxSuppression>& nms4) {
        NMSAttributes attrs;

        attrs.box_encoding = ::ngraph::opset5::NonMaxSuppression::BoxEncodingType::CORNER;
        attrs.is_supported_nms = true;
        attrs.sort_result_descending = true;
        attrs.output_type = ::ngraph::element::i64;

        switch (nms4->get_box_encoding()) {
            case ::ngraph::opset4::NonMaxSuppression::BoxEncodingType::CENTER:
                attrs.box_encoding = ::ngraph::opset5::NonMaxSuppression::BoxEncodingType::CENTER;
                break;
            case ::ngraph::opset4::NonMaxSuppression::BoxEncodingType::CORNER:
                attrs.box_encoding = ::ngraph::opset5::NonMaxSuppression::BoxEncodingType::CORNER;
                break;
            default:
                throw ngraph_error("NonMaxSuppression layer " + nms4->get_friendly_name() +
                                   " has unsupported box encoding");
        }

        attrs.sort_result_descending = nms4->get_sort_result_descending(),
        attrs.output_type = nms4->get_output_type();

        return attrs;
    }

    NMSAttributes get_nms3_attrs(const std::shared_ptr<ngraph::opset3::NonMaxSuppression>& nms3) {
        NMSAttributes attrs;

        attrs.box_encoding = ::ngraph::opset5::NonMaxSuppression::BoxEncodingType::CORNER;
        attrs.is_supported_nms = true;
        attrs.sort_result_descending = true;
        attrs.output_type = ::ngraph::element::i64;

        switch (nms3->get_box_encoding()) {
            case ::ngraph::opset3::NonMaxSuppression::BoxEncodingType::CENTER:
                attrs.box_encoding = ::ngraph::opset5::NonMaxSuppression::BoxEncodingType::CENTER;
                break;
            case ::ngraph::opset3::NonMaxSuppression::BoxEncodingType::CORNER:
                attrs.box_encoding = ::ngraph::opset5::NonMaxSuppression::BoxEncodingType::CORNER;
                break;
            default:
                throw ngraph_error("NonMaxSuppression layer " + nms3->get_friendly_name() +
                                   " has unsupported box encoding");
        }

        attrs.sort_result_descending = nms3->get_sort_result_descending(),
        attrs.output_type = nms3->get_output_type();

        return attrs;
    }

    NMSAttributes get_nms1_attrs(const std::shared_ptr<ngraph::opset1::NonMaxSuppression>& nms1) {
        NMSAttributes attrs;

        attrs.box_encoding = ::ngraph::opset5::NonMaxSuppression::BoxEncodingType::CORNER;
        attrs.is_supported_nms = true;
        attrs.sort_result_descending = true;
        attrs.output_type = ::ngraph::element::i64;

        switch (nms1->get_box_encoding()) {
            case ::ngraph::opset1::NonMaxSuppression::BoxEncodingType::CENTER:
                attrs.box_encoding = ::ngraph::opset5::NonMaxSuppression::BoxEncodingType::CENTER;
                break;
            case ::ngraph::opset1::NonMaxSuppression::BoxEncodingType::CORNER:
                attrs.box_encoding = ::ngraph::opset5::NonMaxSuppression::BoxEncodingType::CORNER;
                break;
            default:
                throw ngraph_error("NonMaxSuppression layer " + nms1->get_friendly_name() +
                                   " has unsupported box encoding");
        }

        attrs.sort_result_descending = nms1->get_sort_result_descending(),

        return attrs;
    }

    NMSAttributes get_nms_attrs(const std::shared_ptr<ngraph::Node>& root) {
        NMSAttributes attrs;
        attrs.output_type = ::ngraph::element::i64;
        attrs.box_encoding = ::ngraph::opset5::NonMaxSuppression::BoxEncodingType::CORNER;
        attrs.sort_result_descending = false;
        attrs.is_supported_nms = false;

        auto nms_4 = std::dynamic_pointer_cast<ngraph::opset4::NonMaxSuppression>(m.get_match_root());
        if (nms_4) {
            return get_nms4_attrs(nms_4);
        }
        auto nms_3 = std::dynamic_pointer_cast<ngraph::opset3::NonMaxSuppression>(m.get_match_root());
        if (nms_4) {
            return get_nms3_attrs(nms_3);
        }
        auto nms_1 = std::dynamic_pointer_cast<ngraph::opset1::NonMaxSuppression>(m.get_match_root());
        if (nms_1) {
            return get_nms1_attrs(nms_3);
        }

        return attrs;
    }

    bool callback_func(pattern::Matcher &m) {
        auto root = m.get_match_root();

        auto attrs = get_nms_attrs(root);
        if (!attrs.is_supported_nms) {
            return false;
        }

        const auto new_args = root->input_values();

        size_t num_of_args = new_args.size();

        const auto& arg2 = num_of_args > 2 ? new_args.at(2) : ngraph::opset5::Constant::create(element::i64, Shape{}, {0});
        const auto& arg3 = num_of_args > 3 ? new_args.at(3) : ngraph::opset5::Constant::create(element::f32, Shape{}, {.0f});
        const auto& arg4 = num_of_args > 4 ? new_args.at(4) : ngraph::opset5::Constant::create(element::f32, Shape{}, {.0f});
        const auto& arg5 = ngraph::opset5::Constant::create(element::f32, Shape{}, {.0f});

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
                attrs.sort_result_descending,
                attrs.output_type);

        new_ops_list.push_back(nms_5);

        // vector of new nGraph operations
        NodeVector new_ops(new_ops_list.begin(), new_ops_list.end());

        nms_5->set_friendly_name(root->get_friendly_name());
        ngraph::copy_runtime_info(root, new_ops);
        root->output(0).replace(nms_5->output(0));
        return true;
    }
} // namespace

NGRAPH_RTTI_DEFINITION(ngraph::pass::ConvertPreviousNMSToNMS5, "ConvertPreviousNMSToNMS5", 0);

ngraph::pass::ConvertPreviousNMSToNMS5::ConvertPreviousNMSToNMS5() {
}

NGRAPH_RTTI_DEFINITION(ngraph::pass::ConvertNMS4ToNMS5, "ConvertNMS4ToNMS5", 0);

ngraph::pass::ConvertNMS4ToNMS5::ConvertNMS4ToNMS5() {
    auto nms = nms4_pattern();
//     auto boxes = std::make_shared<pattern::op::Label>(element::f32, Shape{1, 1000, 4});
//     auto scores = std::make_shared<pattern::op::Label>(element::f32, Shape{1, 1, 1000});
//     auto max_output_boxes_per_class = ngraph::opset4::Constant::create(element::i64, Shape{}, {10});
//     auto iou_threshold = ngraph::opset4::Constant::create(element::f32, Shape{}, {0.75});
//     auto score_threshold = ngraph::opset4::Constant::create(element::f32, Shape{}, {0.7});
//     auto nms = std::make_shared<ngraph::opset4::NonMaxSuppression>(boxes, scores, max_output_boxes_per_class,
//                                                                    iou_threshold, score_threshold);

    ngraph::matcher_pass_callback callback = [](pattern::Matcher &m) {
        auto nms_4 = std::dynamic_pointer_cast<ngraph::opset4::NonMaxSuppression>(m.get_match_root());
        if (!nms_4) {
            return false;
        }

        const auto new_args = nms_4->input_values();

        size_t num_of_args = new_args.size();

        const auto& arg2 = num_of_args > 2 ? new_args.at(2) : ngraph::opset5::Constant::create(element::i64, Shape{}, {0});
        const auto& arg3 = num_of_args > 3 ? new_args.at(3) : ngraph::opset5::Constant::create(element::f32, Shape{}, {.0f});
        const auto& arg4 = num_of_args > 4 ? new_args.at(4) : ngraph::opset5::Constant::create(element::f32, Shape{}, {.0f});
        const auto& arg5 = ngraph::opset5::Constant::create(element::f32, Shape{}, {.0f});

        auto box_encoding = ::ngraph::opset5::NonMaxSuppression::BoxEncodingType::CENTER;
        switch (nms_4->get_box_encoding()) {
            case ::ngraph::opset4::NonMaxSuppression::BoxEncodingType::CENTER:
                box_encoding = ::ngraph::opset5::NonMaxSuppression::BoxEncodingType::CENTER;
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
        nms_4->output(0).replace(nms_5->output(0));
        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(nms, "ConvertNMS4ToNMS5");
    this->register_matcher(m, callback);
}


NGRAPH_RTTI_DEFINITION(ngraph::pass::ConvertNMS3ToNMS5, "ConvertNMS3ToNMS5", 0);

ngraph::pass::ConvertNMS3ToNMS5::ConvertNMS3ToNMS5() {
    auto nms = nms3_pattern();
//     auto boxes = std::make_shared<pattern::op::Label>(element::f32, Shape{1, 1000, 4});
//     auto scores = std::make_shared<pattern::op::Label>(element::f32, Shape{1, 1, 1000});
//     auto max_output_boxes_per_class = ngraph::opset3::Constant::create(element::i64, Shape{}, {10});
//     auto iou_threshold = ngraph::opset3::Constant::create(element::f32, Shape{}, {0.75});
//     auto score_threshold = ngraph::opset3::Constant::create(element::f32, Shape{}, {0.7});
//     auto nms = std::make_shared<ngraph::opset3::NonMaxSuppression>(boxes, scores, max_output_boxes_per_class,
//                                                                    iou_threshold, score_threshold);

    ngraph::matcher_pass_callback callback = [](pattern::Matcher &m) {
        auto nms_3 = std::dynamic_pointer_cast<ngraph::opset3::NonMaxSuppression>(m.get_match_root());
        if (!nms_3) {
            return false;
        }

        const auto new_args = nms_3->input_values();

        size_t num_of_args = new_args.size();

        const auto& arg2 = num_of_args > 2 ? new_args.at(2) : ngraph::opset5::Constant::create(element::i64, Shape{}, {0});
        const auto& arg3 = num_of_args > 3 ? new_args.at(3) : ngraph::opset5::Constant::create(element::f32, Shape{}, {.0f});
        const auto& arg4 = num_of_args > 4 ? new_args.at(4) : ngraph::opset5::Constant::create(element::f32, Shape{}, {.0f});
        const auto& arg5 = ngraph::opset5::Constant::create(element::f32, Shape{}, {.0f});

        auto box_encoding = ::ngraph::opset5::NonMaxSuppression::BoxEncodingType::CENTER;
        switch (nms_3->get_box_encoding()) {
            case ::ngraph::opset3::NonMaxSuppression::BoxEncodingType::CENTER:
                box_encoding = ::ngraph::opset5::NonMaxSuppression::BoxEncodingType::CENTER;
                break;
            case ::ngraph::opset3::NonMaxSuppression::BoxEncodingType::CORNER:
                box_encoding = ::ngraph::opset5::NonMaxSuppression::BoxEncodingType::CORNER;
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
        nms_3->output(0).replace(nms_5->output(0));
        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(nms, "ConvertNMS3ToNMS5");
    this->register_matcher(m, callback);
}

NGRAPH_RTTI_DEFINITION(ngraph::pass::ConvertNMS1ToNMS5, "ConvertNMS1ToNMS5", 0);

ngraph::pass::ConvertNMS1ToNMS5::ConvertNMS1ToNMS5() {
    auto nms = nms1_pattern();
//     auto boxes = std::make_shared<pattern::op::Label>(element::f32, Shape{1, 1000, 4});
//     auto scores = std::make_shared<pattern::op::Label>(element::f32, Shape{1, 1, 1000});
//     auto max_output_boxes_per_class = ngraph::opset1::Constant::create(element::i64, Shape{}, {10});
//     auto iou_threshold = ngraph::opset1::Constant::create(element::f32, Shape{}, {0.75});
//     auto score_threshold = ngraph::opset1::Constant::create(element::f32, Shape{}, {0.7});
//     auto nms = std::make_shared<ngraph::opset1::NonMaxSuppression>(boxes, scores, max_output_boxes_per_class,
//                                                                    iou_threshold, score_threshold);

    ngraph::matcher_pass_callback callback = [](pattern::Matcher &m) {
        auto nms_1 = std::dynamic_pointer_cast<ngraph::opset1::NonMaxSuppression>(m.get_match_root());
        if (!nms_1) {
            return false;
        }

        const auto new_args = nms_1->input_values();

        size_t num_of_args = new_args.size();

        const auto& arg2 = num_of_args > 2 ? new_args.at(2) : ngraph::opset5::Constant::create(element::i32, Shape{}, {0});
        const auto& arg3 = num_of_args > 3 ? new_args.at(3) : ngraph::opset5::Constant::create(element::f32, Shape{}, {.0f});
        const auto& arg4 = num_of_args > 4 ? new_args.at(4) : ngraph::opset5::Constant::create(element::f32, Shape{}, {.0f});
        const auto& arg5 = ngraph::opset5::Constant::create(element::f32, Shape{}, {.0f});

        auto box_encoding = ::ngraph::opset5::NonMaxSuppression::BoxEncodingType::CENTER;
        switch (nms_1->get_box_encoding()) {
            case ::ngraph::opset1::NonMaxSuppression::BoxEncodingType::CENTER:
                box_encoding = ::ngraph::opset5::NonMaxSuppression::BoxEncodingType::CENTER;
                break;
            case ::ngraph::opset1::NonMaxSuppression::BoxEncodingType::CORNER:
                box_encoding = ::ngraph::opset5::NonMaxSuppression::BoxEncodingType::CORNER;
                break;
            default:
                throw ngraph_error("NonMaxSuppression layer " + nms_1->get_friendly_name() +
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
                nms_1->get_sort_result_descending(),
                ::ngraph::element::i64);

        new_ops_list.push_back(nms_5);

        // vector of new nGraph operations
        NodeVector new_ops(new_ops_list.begin(), new_ops_list.end());

        nms_5->set_friendly_name(nms_1->get_friendly_name());
        ngraph::copy_runtime_info(nms_1, new_ops);
        nms_1->output(0).replace(nms_5->output(0));
        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(nms, "ConvertNMS1ToNMS5");
    this->register_matcher(m, callback);
}
