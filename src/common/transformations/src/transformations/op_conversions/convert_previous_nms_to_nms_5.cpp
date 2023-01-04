// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/op_conversions/convert_previous_nms_to_nms_5.hpp"

#include <list>
#include <memory>
#include <ngraph/rt_info.hpp>
#include <openvino/opsets/opset1.hpp>
#include <openvino/opsets/opset3.hpp>
#include <openvino/opsets/opset4.hpp>
#include <openvino/opsets/opset5.hpp>
#include <openvino/pass/pattern/op/wrap_type.hpp>
#include <vector>

#include "itt.hpp"

using namespace ov;

namespace {
struct NMSAttributes {
    ngraph::element::Type output_type;
    ov::opset5::NonMaxSuppression::BoxEncodingType box_encoding;
    bool sort_result_descending;
    bool is_supported_nms;
};

NMSAttributes get_nms4_attrs(const std::shared_ptr<ov::opset4::NonMaxSuppression>& nms4) {
    NMSAttributes attrs;

    attrs.box_encoding = ::ov::opset5::NonMaxSuppression::BoxEncodingType::CORNER;
    attrs.is_supported_nms = true;
    attrs.sort_result_descending = true;
    attrs.output_type = ::ngraph::element::i64;

    switch (nms4->get_box_encoding()) {
    case ::ov::opset4::NonMaxSuppression::BoxEncodingType::CENTER:
        attrs.box_encoding = ::ov::opset5::NonMaxSuppression::BoxEncodingType::CENTER;
        break;
    case ::ov::opset4::NonMaxSuppression::BoxEncodingType::CORNER:
        attrs.box_encoding = ::ov::opset5::NonMaxSuppression::BoxEncodingType::CORNER;
        break;
    default:
        throw Exception("NonMaxSuppression layer " + nms4->get_friendly_name() + " has unsupported box encoding");
    }

    attrs.sort_result_descending = nms4->get_sort_result_descending();
    attrs.output_type = nms4->get_output_type();

    return attrs;
}

NMSAttributes get_nms3_attrs(const std::shared_ptr<ov::opset3::NonMaxSuppression>& nms3) {
    NMSAttributes attrs;

    attrs.box_encoding = ::ov::opset5::NonMaxSuppression::BoxEncodingType::CORNER;
    attrs.is_supported_nms = true;
    attrs.sort_result_descending = true;
    attrs.output_type = ::ngraph::element::i64;

    switch (nms3->get_box_encoding()) {
    case ::ov::opset3::NonMaxSuppression::BoxEncodingType::CENTER:
        attrs.box_encoding = ::ov::opset5::NonMaxSuppression::BoxEncodingType::CENTER;
        break;
    case ::ov::opset3::NonMaxSuppression::BoxEncodingType::CORNER:
        attrs.box_encoding = ::ov::opset5::NonMaxSuppression::BoxEncodingType::CORNER;
        break;
    default:
        throw Exception("NonMaxSuppression layer " + nms3->get_friendly_name() + " has unsupported box encoding");
    }

    attrs.sort_result_descending = nms3->get_sort_result_descending();
    attrs.output_type = nms3->get_output_type();

    return attrs;
}

NMSAttributes get_nms1_attrs(const std::shared_ptr<ov::opset1::NonMaxSuppression>& nms1) {
    NMSAttributes attrs;

    attrs.box_encoding = ::ov::opset5::NonMaxSuppression::BoxEncodingType::CORNER;
    attrs.is_supported_nms = true;
    attrs.sort_result_descending = true;
    attrs.output_type = ::ngraph::element::i64;

    switch (nms1->get_box_encoding()) {
    case ::ov::opset1::NonMaxSuppression::BoxEncodingType::CENTER:
        attrs.box_encoding = ::ov::opset5::NonMaxSuppression::BoxEncodingType::CENTER;
        break;
    case ::ov::opset1::NonMaxSuppression::BoxEncodingType::CORNER:
        attrs.box_encoding = ::ov::opset5::NonMaxSuppression::BoxEncodingType::CORNER;
        break;
    default:
        throw Exception("NonMaxSuppression layer " + nms1->get_friendly_name() + " has unsupported box encoding");
    }

    attrs.sort_result_descending = nms1->get_sort_result_descending();

    return attrs;
}

NMSAttributes get_nms_attrs(const std::shared_ptr<ngraph::Node>& root) {
    NMSAttributes attrs;
    attrs.output_type = ::ngraph::element::i64;
    attrs.box_encoding = ::ov::opset5::NonMaxSuppression::BoxEncodingType::CORNER;
    attrs.sort_result_descending = false;
    attrs.is_supported_nms = false;

    auto nms_4 = std::dynamic_pointer_cast<ov::opset4::NonMaxSuppression>(root);
    if (nms_4) {
        return get_nms4_attrs(nms_4);
    }
    auto nms_3 = std::dynamic_pointer_cast<ov::opset3::NonMaxSuppression>(root);
    if (nms_3) {
        return get_nms3_attrs(nms_3);
    }
    auto nms_1 = std::dynamic_pointer_cast<ov::opset1::NonMaxSuppression>(root);
    if (nms_1) {
        return get_nms1_attrs(nms_1);
    }

    return attrs;
}

bool callback_func(pass::pattern::Matcher& m, pass::MatcherPass* impl) {
    auto root = m.get_match_root();

    auto attrs = get_nms_attrs(root);
    if (!attrs.is_supported_nms) {
        return false;
    }

    const auto new_args = root->input_values();

    size_t num_of_args = new_args.size();

    const auto& arg2 = num_of_args > 2 ? new_args.at(2) : ov::opset5::Constant::create(element::i64, Shape{}, {0});
    const auto& arg3 = num_of_args > 3 ? new_args.at(3) : ov::opset5::Constant::create(element::f32, Shape{}, {.0f});
    const auto& arg4 = num_of_args > 4 ? new_args.at(4) : ov::opset5::Constant::create(element::f32, Shape{}, {.0f});

    const auto nms_5 = impl->register_new_node<opset5::NonMaxSuppression>(new_args.at(0),
                                                                          new_args.at(1),
                                                                          arg2,
                                                                          arg3,
                                                                          arg4,
                                                                          attrs.box_encoding,
                                                                          attrs.sort_result_descending,
                                                                          attrs.output_type);

    nms_5->set_friendly_name(root->get_friendly_name());
    ngraph::copy_runtime_info(root, nms_5);
    root->output(0).replace(nms_5->output(0));
    return true;
}
}  // namespace

ov::pass::ConvertNMS4ToNMS5::ConvertNMS4ToNMS5() {
    MATCHER_SCOPE(ConvertNMS4ToNMS5);
    auto nms = pattern::wrap_type<ov::opset4::NonMaxSuppression>();
    matcher_pass_callback callback = [this](pattern::Matcher& m) {
        return callback_func(m, this);
    };

    auto m = std::make_shared<pattern::Matcher>(nms, matcher_name);
    this->register_matcher(m, callback);
}

ov::pass::ConvertNMS3ToNMS5::ConvertNMS3ToNMS5() {
    MATCHER_SCOPE(ConvertNMS3ToNMS5);
    auto nms = pattern::wrap_type<ov::opset3::NonMaxSuppression>();
    matcher_pass_callback callback = [this](pattern::Matcher& m) {
        return callback_func(m, this);
    };

    auto m = std::make_shared<pattern::Matcher>(nms, matcher_name);
    this->register_matcher(m, callback);
}

ov::pass::ConvertNMS1ToNMS5::ConvertNMS1ToNMS5() {
    MATCHER_SCOPE(ConvertNMS1ToNMS5);
    auto nms = pattern::wrap_type<ov::opset1::NonMaxSuppression>();
    matcher_pass_callback callback = [this](pattern::Matcher& m) {
        return callback_func(m, this);
    };

    auto m = std::make_shared<pattern::Matcher>(nms, matcher_name);
    this->register_matcher(m, callback);
}
