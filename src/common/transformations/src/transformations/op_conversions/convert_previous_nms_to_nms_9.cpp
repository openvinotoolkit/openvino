// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/op_conversions/convert_previous_nms_to_nms_9.hpp"

#include <list>
#include <memory>
#include <ngraph/opsets/opset1.hpp>
#include <ngraph/opsets/opset3.hpp>
#include <ngraph/opsets/opset4.hpp>
#include <ngraph/opsets/opset5.hpp>
#include <ngraph/opsets/opset9.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>
#include <ngraph/rt_info.hpp>
#include <vector>

#include "itt.hpp"

using namespace ngraph;

namespace {
struct NMS9Attributes {
    ngraph::element::Type output_type;
    ngraph::opset9::NonMaxSuppression::BoxEncodingType box_encoding;
    bool sort_result_descending;
    bool is_supported_nms;
};

NMS9Attributes get_nms9_attrs_from_nms5(const std::shared_ptr<ngraph::opset5::NonMaxSuppression>& nms5) {
    NMS9Attributes attrs;

    attrs.box_encoding = ::ngraph::opset9::NonMaxSuppression::BoxEncodingType::CORNER;
    attrs.is_supported_nms = true;
    attrs.sort_result_descending = true;
    attrs.output_type = ::ngraph::element::i64;

    switch (nms5->get_box_encoding()) {
    case ::ngraph::opset5::NonMaxSuppression::BoxEncodingType::CENTER:
        attrs.box_encoding = ::ngraph::opset9::NonMaxSuppression::BoxEncodingType::CENTER;
        break;
    case ::ngraph::opset5::NonMaxSuppression::BoxEncodingType::CORNER:
        attrs.box_encoding = ::ngraph::opset9::NonMaxSuppression::BoxEncodingType::CORNER;
        break;
    default:
        throw ngraph_error("NonMaxSuppression layer " + nms5->get_friendly_name() + " has unsupported box encoding");
    }

    attrs.sort_result_descending = nms5->get_sort_result_descending();
    attrs.output_type = nms5->get_output_type();

    return attrs;
}

NMS9Attributes get_nms9_attrs_from_nms4(const std::shared_ptr<ngraph::opset4::NonMaxSuppression>& nms4) {
    NMS9Attributes attrs;

    attrs.box_encoding = ::ngraph::opset9::NonMaxSuppression::BoxEncodingType::CORNER;
    attrs.is_supported_nms = true;
    attrs.sort_result_descending = true;
    attrs.output_type = ::ngraph::element::i64;

    switch (nms4->get_box_encoding()) {
    case ::ngraph::opset4::NonMaxSuppression::BoxEncodingType::CENTER:
        attrs.box_encoding = ::ngraph::opset9::NonMaxSuppression::BoxEncodingType::CENTER;
        break;
    case ::ngraph::opset4::NonMaxSuppression::BoxEncodingType::CORNER:
        attrs.box_encoding = ::ngraph::opset9::NonMaxSuppression::BoxEncodingType::CORNER;
        break;
    default:
        throw ngraph_error("NonMaxSuppression layer " + nms4->get_friendly_name() + " has unsupported box encoding");
    }

    attrs.sort_result_descending = nms4->get_sort_result_descending();
    attrs.output_type = nms4->get_output_type();

    return attrs;
}

NMS9Attributes get_nms9_attrs_from_nms3(const std::shared_ptr<ngraph::opset3::NonMaxSuppression>& nms3) {
    NMS9Attributes attrs;

    attrs.box_encoding = ::ngraph::opset9::NonMaxSuppression::BoxEncodingType::CORNER;
    attrs.is_supported_nms = true;
    attrs.sort_result_descending = true;
    attrs.output_type = ::ngraph::element::i64;

    switch (nms3->get_box_encoding()) {
    case ::ngraph::opset3::NonMaxSuppression::BoxEncodingType::CENTER:
        attrs.box_encoding = ::ngraph::opset9::NonMaxSuppression::BoxEncodingType::CENTER;
        break;
    case ::ngraph::opset3::NonMaxSuppression::BoxEncodingType::CORNER:
        attrs.box_encoding = ::ngraph::opset9::NonMaxSuppression::BoxEncodingType::CORNER;
        break;
    default:
        throw ngraph_error("NonMaxSuppression layer " + nms3->get_friendly_name() + " has unsupported box encoding");
    }

    attrs.sort_result_descending = nms3->get_sort_result_descending();
    attrs.output_type = nms3->get_output_type();

    return attrs;
}

NMS9Attributes get_nms9_attrs_from_nms1(const std::shared_ptr<ngraph::opset1::NonMaxSuppression>& nms1) {
    NMS9Attributes attrs;

    attrs.box_encoding = ::ngraph::opset9::NonMaxSuppression::BoxEncodingType::CORNER;
    attrs.is_supported_nms = true;
    attrs.sort_result_descending = true;
    attrs.output_type = ::ngraph::element::i64;

    switch (nms1->get_box_encoding()) {
    case ::ngraph::opset1::NonMaxSuppression::BoxEncodingType::CENTER:
        attrs.box_encoding = ::ngraph::opset9::NonMaxSuppression::BoxEncodingType::CENTER;
        break;
    case ::ngraph::opset1::NonMaxSuppression::BoxEncodingType::CORNER:
        attrs.box_encoding = ::ngraph::opset9::NonMaxSuppression::BoxEncodingType::CORNER;
        break;
    default:
        throw ngraph_error("NonMaxSuppression layer " + nms1->get_friendly_name() + " has unsupported box encoding");
    }

    attrs.sort_result_descending = nms1->get_sort_result_descending();

    return attrs;
}

NMS9Attributes get_nms9_attrs(const std::shared_ptr<ngraph::Node>& root) {
    NMS9Attributes attrs;
    attrs.output_type = ::ngraph::element::i64;
    attrs.box_encoding = ::ngraph::opset9::NonMaxSuppression::BoxEncodingType::CORNER;
    attrs.sort_result_descending = false;
    attrs.is_supported_nms = false;

    auto nms_5 = std::dynamic_pointer_cast<ngraph::opset5::NonMaxSuppression>(root);
    if (nms_5) {
        return get_nms9_attrs_from_nms5(nms_5);
    }
    auto nms_4 = std::dynamic_pointer_cast<ngraph::opset4::NonMaxSuppression>(root);
    if (nms_4) {
        return get_nms9_attrs_from_nms4(nms_4);
    }
    auto nms_3 = std::dynamic_pointer_cast<ngraph::opset3::NonMaxSuppression>(root);
    if (nms_3) {
        return get_nms9_attrs_from_nms3(nms_3);
    }
    auto nms_1 = std::dynamic_pointer_cast<ngraph::opset1::NonMaxSuppression>(root);
    if (nms_1) {
        return get_nms9_attrs_from_nms1(nms_1);
    }

    return attrs;
}

bool nms_to_nms9_callback_func(pattern::Matcher& m, pass::MatcherPass* impl) {
    auto root = m.get_match_root();

    auto attrs = get_nms9_attrs(root);
    if (!attrs.is_supported_nms) {
        return false;
    }

    const auto nms_input = root->input_values();

    size_t num_of_args = nms_input.size();

    const auto& max_selected_box =
        num_of_args > 2 ? nms_input.at(2) : ngraph::opset1::Constant::create(element::i64, Shape{}, {0});
    const auto& iou_threshold =
        num_of_args > 3 ? nms_input.at(3) : ngraph::opset1::Constant::create(element::f32, Shape{}, {.0f});
    const auto& score_threshold =
        num_of_args > 4 ? nms_input.at(4) : ngraph::opset1::Constant::create(element::f32, Shape{}, {.0f});
    const auto& soft_sigma =
        num_of_args > 5 ? nms_input.at(5) : ngraph::opset1::Constant::create(element::f32, Shape{}, {.0f});

    const auto nms_9 = impl->register_new_node<ngraph::op::v9::NonMaxSuppression>(nms_input.at(0),
                                                                                  nms_input.at(1),
                                                                                  max_selected_box,
                                                                                  iou_threshold,
                                                                                  score_threshold,
                                                                                  soft_sigma,
                                                                                  attrs.box_encoding,
                                                                                  attrs.sort_result_descending,
                                                                                  attrs.output_type);

    nms_9->set_friendly_name(root->get_friendly_name());
    ngraph::copy_runtime_info(root, nms_9);
    root->output(0).replace(nms_9->output(0));
    return true;
}
}  // namespace

ngraph::pass::ConvertNMS5ToNMS9::ConvertNMS5ToNMS9() {
    MATCHER_SCOPE(ConvertNMS5ToNMS9);
    auto nms = ngraph::pattern::wrap_type<ngraph::opset5::NonMaxSuppression>();
    ngraph::matcher_pass_callback callback = [this](pattern::Matcher& m) {
        return nms_to_nms9_callback_func(m, this);
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(nms, matcher_name);
    this->register_matcher(m, callback);
}

ngraph::pass::ConvertNMS4ToNMS9::ConvertNMS4ToNMS9() {
    MATCHER_SCOPE(ConvertNMS4ToNMS9);
    auto nms = ngraph::pattern::wrap_type<ngraph::opset4::NonMaxSuppression>();
    ngraph::matcher_pass_callback callback = [this](pattern::Matcher& m) {
        return nms_to_nms9_callback_func(m, this);
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(nms, matcher_name);
    this->register_matcher(m, callback);
}

ngraph::pass::ConvertNMS3ToNMS9::ConvertNMS3ToNMS9() {
    MATCHER_SCOPE(ConvertNMS3ToNMS9);
    auto nms = ngraph::pattern::wrap_type<ngraph::opset3::NonMaxSuppression>();
    ngraph::matcher_pass_callback callback = [this](pattern::Matcher& m) {
        return nms_to_nms9_callback_func(m, this);
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(nms, matcher_name);
    this->register_matcher(m, callback);
}

ngraph::pass::ConvertNMS1ToNMS9::ConvertNMS1ToNMS9() {
    MATCHER_SCOPE(ConvertNMS1ToNMS9);
    auto nms = ngraph::pattern::wrap_type<ngraph::opset1::NonMaxSuppression>();
    ngraph::matcher_pass_callback callback = [this](pattern::Matcher& m) {
        return nms_to_nms9_callback_func(m, this);
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(nms, matcher_name);
    this->register_matcher(m, callback);
}
