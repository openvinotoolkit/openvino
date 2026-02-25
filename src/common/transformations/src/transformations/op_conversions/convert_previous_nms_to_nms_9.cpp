// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/op_conversions/convert_previous_nms_to_nms_9.hpp"

#include <list>
#include <memory>
#include <vector>

#include "itt.hpp"
#include "openvino/core/graph_util.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/non_max_suppression.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"

using namespace ov;

namespace v0 = ov::op::v0;
namespace v1 = ov::op::v1;
namespace v3 = ov::op::v3;
namespace v4 = ov::op::v4;
namespace v5 = ov::op::v5;
namespace v9 = ov::op::v9;

namespace ov::pass {

namespace {
struct NMS9Attributes {
    ov::element::Type output_type;
    v9::NonMaxSuppression::BoxEncodingType box_encoding;
    bool sort_result_descending;
    bool is_supported_nms;
};

NMS9Attributes get_nms9_attrs_from_nms5(const std::shared_ptr<v5::NonMaxSuppression>& nms5) {
    NMS9Attributes attrs;

    attrs.box_encoding = v9::NonMaxSuppression::BoxEncodingType::CORNER;
    attrs.is_supported_nms = true;
    attrs.sort_result_descending = true;
    attrs.output_type = ::ov::element::i64;

    switch (nms5->get_box_encoding()) {
    case v5::NonMaxSuppression::BoxEncodingType::CENTER:
        attrs.box_encoding = v9::NonMaxSuppression::BoxEncodingType::CENTER;
        break;
    case v5::NonMaxSuppression::BoxEncodingType::CORNER:
        attrs.box_encoding = v9::NonMaxSuppression::BoxEncodingType::CORNER;
        break;
    default:
        OPENVINO_THROW("NonMaxSuppression layer " + nms5->get_friendly_name() + " has unsupported box encoding");
    }

    attrs.sort_result_descending = nms5->get_sort_result_descending();
    attrs.output_type = nms5->get_output_type();

    return attrs;
}

NMS9Attributes get_nms9_attrs_from_nms4(const std::shared_ptr<v4::NonMaxSuppression>& nms4) {
    NMS9Attributes attrs;

    attrs.box_encoding = v9::NonMaxSuppression::BoxEncodingType::CORNER;
    attrs.is_supported_nms = true;
    attrs.sort_result_descending = true;
    attrs.output_type = ::ov::element::i64;

    switch (nms4->get_box_encoding()) {
    case v4::NonMaxSuppression::BoxEncodingType::CENTER:
        attrs.box_encoding = v9::NonMaxSuppression::BoxEncodingType::CENTER;
        break;
    case v4::NonMaxSuppression::BoxEncodingType::CORNER:
        attrs.box_encoding = v9::NonMaxSuppression::BoxEncodingType::CORNER;
        break;
    default:
        OPENVINO_THROW("NonMaxSuppression layer " + nms4->get_friendly_name() + " has unsupported box encoding");
    }

    attrs.sort_result_descending = nms4->get_sort_result_descending();
    attrs.output_type = nms4->get_output_type();

    return attrs;
}

NMS9Attributes get_nms9_attrs_from_nms3(const std::shared_ptr<v3::NonMaxSuppression>& nms3) {
    NMS9Attributes attrs;

    attrs.box_encoding = v9::NonMaxSuppression::BoxEncodingType::CORNER;
    attrs.is_supported_nms = true;
    attrs.sort_result_descending = true;
    attrs.output_type = ::ov::element::i64;

    switch (nms3->get_box_encoding()) {
    case v3::NonMaxSuppression::BoxEncodingType::CENTER:
        attrs.box_encoding = v9::NonMaxSuppression::BoxEncodingType::CENTER;
        break;
    case v3::NonMaxSuppression::BoxEncodingType::CORNER:
        attrs.box_encoding = v9::NonMaxSuppression::BoxEncodingType::CORNER;
        break;
    default:
        OPENVINO_THROW("NonMaxSuppression layer " + nms3->get_friendly_name() + " has unsupported box encoding");
    }

    attrs.sort_result_descending = nms3->get_sort_result_descending();
    attrs.output_type = nms3->get_output_type();

    return attrs;
}

NMS9Attributes get_nms9_attrs_from_nms1(const std::shared_ptr<v1::NonMaxSuppression>& nms1) {
    NMS9Attributes attrs;

    attrs.box_encoding = v9::NonMaxSuppression::BoxEncodingType::CORNER;
    attrs.is_supported_nms = true;
    attrs.sort_result_descending = true;
    attrs.output_type = ::ov::element::i64;

    switch (nms1->get_box_encoding()) {
    case v1::NonMaxSuppression::BoxEncodingType::CENTER:
        attrs.box_encoding = v9::NonMaxSuppression::BoxEncodingType::CENTER;
        break;
    case v1::NonMaxSuppression::BoxEncodingType::CORNER:
        attrs.box_encoding = v9::NonMaxSuppression::BoxEncodingType::CORNER;
        break;
    default:
        OPENVINO_THROW("NonMaxSuppression layer " + nms1->get_friendly_name() + " has unsupported box encoding");
    }

    attrs.sort_result_descending = nms1->get_sort_result_descending();

    return attrs;
}

NMS9Attributes get_nms9_attrs(const std::shared_ptr<ov::Node>& root) {
    NMS9Attributes attrs;
    attrs.output_type = ::ov::element::i64;
    attrs.box_encoding = v9::NonMaxSuppression::BoxEncodingType::CORNER;
    attrs.sort_result_descending = false;
    attrs.is_supported_nms = false;

    auto nms_5 = ov::as_type_ptr<v5::NonMaxSuppression>(root);
    if (nms_5) {
        return get_nms9_attrs_from_nms5(nms_5);
    }
    auto nms_4 = ov::as_type_ptr<v4::NonMaxSuppression>(root);
    if (nms_4) {
        return get_nms9_attrs_from_nms4(nms_4);
    }
    auto nms_3 = ov::as_type_ptr<v3::NonMaxSuppression>(root);
    if (nms_3) {
        return get_nms9_attrs_from_nms3(nms_3);
    }
    auto nms_1 = ov::as_type_ptr<v1::NonMaxSuppression>(root);
    if (nms_1) {
        return get_nms9_attrs_from_nms1(nms_1);
    }

    return attrs;
}

bool nms_to_nms9_callback_func(pattern::Matcher& m, MatcherPass* impl) {
    auto root = m.get_match_root();

    auto attrs = get_nms9_attrs(root);
    if (!attrs.is_supported_nms) {
        return false;
    }

    const auto nms_input = root->input_values();

    size_t num_of_args = nms_input.size();

    const auto& max_selected_box = num_of_args > 2 ? nms_input.at(2) : v0::Constant::create(element::i64, Shape{}, {0});
    const auto& iou_threshold = num_of_args > 3 ? nms_input.at(3) : v0::Constant::create(element::f32, Shape{}, {.0f});
    const auto& score_threshold =
        num_of_args > 4 ? nms_input.at(4) : v0::Constant::create(element::f32, Shape{}, {.0f});
    const auto& soft_sigma = num_of_args > 5 ? nms_input.at(5) : v0::Constant::create(element::f32, Shape{}, {.0f});

    const auto nms_9 = impl->register_new_node<v9::NonMaxSuppression>(nms_input.at(0),
                                                                      nms_input.at(1),
                                                                      max_selected_box,
                                                                      iou_threshold,
                                                                      score_threshold,
                                                                      soft_sigma,
                                                                      attrs.box_encoding,
                                                                      attrs.sort_result_descending,
                                                                      attrs.output_type);

    nms_9->set_friendly_name(root->get_friendly_name());
    ov::copy_runtime_info(root, nms_9);
    // nms0-4 have one output, nms5/9 have 3 outputs.
    if (ov::as_type_ptr<v5::NonMaxSuppression>(root))
        ov::replace_node(root, nms_9);
    else
        root->output(0).replace(nms_9->output(0));
    return true;
}
}  // namespace

ConvertNMS5ToNMS9::ConvertNMS5ToNMS9() {
    MATCHER_SCOPE(ConvertNMS5ToNMS9);
    auto nms = pattern::wrap_type<v5::NonMaxSuppression>();
    matcher_pass_callback callback = [this](pattern::Matcher& m) {
        return nms_to_nms9_callback_func(m, this);
    };

    auto m = std::make_shared<pattern::Matcher>(nms, matcher_name);
    this->register_matcher(m, callback);
}

ConvertNMS4ToNMS9::ConvertNMS4ToNMS9() {
    MATCHER_SCOPE(ConvertNMS4ToNMS9);
    auto nms = pattern::wrap_type<v4::NonMaxSuppression>();
    matcher_pass_callback callback = [this](pattern::Matcher& m) {
        return nms_to_nms9_callback_func(m, this);
    };

    auto m = std::make_shared<pattern::Matcher>(nms, matcher_name);
    this->register_matcher(m, callback);
}

ConvertNMS3ToNMS9::ConvertNMS3ToNMS9() {
    MATCHER_SCOPE(ConvertNMS3ToNMS9);
    auto nms = pattern::wrap_type<v3::NonMaxSuppression>();
    matcher_pass_callback callback = [this](pattern::Matcher& m) {
        return nms_to_nms9_callback_func(m, this);
    };

    auto m = std::make_shared<pattern::Matcher>(nms, matcher_name);
    this->register_matcher(m, callback);
}

ConvertNMS1ToNMS9::ConvertNMS1ToNMS9() {
    MATCHER_SCOPE(ConvertNMS1ToNMS9);
    auto nms = pattern::wrap_type<v1::NonMaxSuppression>();
    matcher_pass_callback callback = [this](pattern::Matcher& m) {
        return nms_to_nms9_callback_func(m, this);
    };

    auto m = std::make_shared<pattern::Matcher>(nms, matcher_name);
    this->register_matcher(m, callback);
}

}  // namespace ov::pass
