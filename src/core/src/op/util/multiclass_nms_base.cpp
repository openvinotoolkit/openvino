// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/util/multiclass_nms_base.hpp"

#include "itt.hpp"

namespace ov {
op::util::MulticlassNmsBase::MulticlassNmsBase(const OutputVector& arguments, const Attributes& attrs)
    : Op(arguments),
      m_attrs{attrs} {}

void op::util::MulticlassNmsBase::validate() {
    OV_OP_SCOPE(util_MulticlassNmsBase_validate);

    const auto& nms_attrs = this->get_attrs();
    const auto output_type = nms_attrs.output_type;
    const auto nms_top_k = nms_attrs.nms_top_k;
    const auto keep_top_k = nms_attrs.keep_top_k;

    auto is_float_type_admissible = [](const element::Type& t) {
        return t == element::f32 || t == element::f16 || t == element::bf16;
    };

    // validate dtype of each input
    NODE_VALIDATION_CHECK(this,
                          output_type == element::i64 || output_type == element::i32,
                          "Output type must be i32 or i64");

    NODE_VALIDATION_CHECK(this,
                          is_float_type_admissible(this->get_input_element_type(0)),
                          "Expected bf16, fp16 or fp32 as element type for the 'boxes' input.");

    NODE_VALIDATION_CHECK(this,
                          is_float_type_admissible(this->get_input_element_type(1)),
                          "Expected bf16, fp16 or fp32 as element type for the 'scores' input.");

    NODE_VALIDATION_CHECK(this,
                          this->get_input_element_type(0).compatible(this->get_input_element_type(1)),
                          "Expected 'boxes', 'scores' type is same.");

    if (this->get_input_size() == 3) {
        NODE_VALIDATION_CHECK(
            this,
            this->get_input_element_type(2) == element::i64 || this->get_input_element_type(2) == element::i32,
            "Expected i64 or i32 as element type for the 'roisnum' input.");
    }

    // validate attributes
    NODE_VALIDATION_CHECK(this, nms_top_k >= -1, "The 'nms_top_k' must be great or equal -1. Got:", nms_top_k);

    NODE_VALIDATION_CHECK(this, keep_top_k >= -1, "The 'keep_top_k' must be great or equal -1. Got:", keep_top_k);

    NODE_VALIDATION_CHECK(this,
                          m_attrs.background_class >= -1,
                          "The 'background_class' must be great or equal -1. Got:",
                          m_attrs.background_class);

    NODE_VALIDATION_CHECK(this,
                          m_attrs.nms_eta >= 0.0f && m_attrs.nms_eta <= 1.0f,
                          "The 'nms_eta' must be in close range [0, 1.0]. Got:",
                          m_attrs.nms_eta);
}

bool op::util::MulticlassNmsBase::visit_attributes(AttributeVisitor& visitor) {
    OV_OP_SCOPE(util_MulticlassNmsBase_visit_attributes);

    visitor.on_attribute("sort_result_type", m_attrs.sort_result_type);
    visitor.on_attribute("output_type", m_attrs.output_type);
    visitor.on_attribute("nms_top_k", m_attrs.nms_top_k);
    visitor.on_attribute("keep_top_k", m_attrs.keep_top_k);
    visitor.on_attribute("sort_result_across_batch", m_attrs.sort_result_across_batch);
    visitor.on_attribute("iou_threshold", m_attrs.iou_threshold);
    visitor.on_attribute("score_threshold", m_attrs.score_threshold);
    visitor.on_attribute("background_class", m_attrs.background_class);
    visitor.on_attribute("nms_eta", m_attrs.nms_eta);
    visitor.on_attribute("normalized", m_attrs.normalized);

    return true;
}

void op::util::MulticlassNmsBase::set_attrs(op::util::MulticlassNmsBase::Attributes attrs) {
    m_attrs = std::move(attrs);
}

std::ostream& operator<<(std::ostream& s, const op::util::MulticlassNmsBase::SortResultType& type) {
    return s << as_string(type);
}

template <>
OPENVINO_API EnumNames<op::util::MulticlassNmsBase::SortResultType>&
EnumNames<op::util::MulticlassNmsBase::SortResultType>::get() {
    static auto enum_names = EnumNames<op::util::MulticlassNmsBase::SortResultType>(
        "op::util::MulticlassNmsBase::SortResultType",
        {{"classid", op::util::MulticlassNmsBase::SortResultType::CLASSID},
         {"score", op::util::MulticlassNmsBase::SortResultType::SCORE},
         {"none", op::util::MulticlassNmsBase::SortResultType::NONE}});
    return enum_names;
}

AttributeAdapter<op::util::MulticlassNmsBase::SortResultType>::~AttributeAdapter() = default;
}  // namespace ov
