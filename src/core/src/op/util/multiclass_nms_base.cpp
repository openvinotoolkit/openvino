// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/util/multiclass_nms_base.hpp"

#include "itt.hpp"

using namespace ov;

BWDCMP_RTTI_DEFINITION(ov::op::util::MulticlassNmsBase);

op::util::MulticlassNmsBase::MulticlassNmsBase()
    : NmsBase(m_attrs.output_type, m_attrs.nms_top_k, m_attrs.keep_top_k) {}

op::util::MulticlassNmsBase::MulticlassNmsBase(const OutputVector& arguments, const Attributes& attrs)
    : NmsBase(arguments, m_attrs.output_type, m_attrs.nms_top_k, m_attrs.keep_top_k),
      m_attrs{attrs} {}

bool op::util::MulticlassNmsBase::validate() {
    NGRAPH_OP_SCOPE(util_MulticlassNmsBase_validate);
    const auto validated = NmsBase::validate();

    NODE_VALIDATION_CHECK(this,
                          m_attrs.background_class >= -1,
                          "The 'background_class' must be great or equal -1. Got:",
                          m_attrs.background_class);

    NODE_VALIDATION_CHECK(this,
                          m_attrs.nms_eta >= 0.0f && m_attrs.nms_eta <= 1.0f,
                          "The 'nms_eta' must be in close range [0, 1.0]. Got:",
                          m_attrs.nms_eta);
    return validated;
}

bool op::util::MulticlassNmsBase::visit_attributes(AttributeVisitor& visitor) {
    NGRAPH_OP_SCOPE(util_MulticlassNmsBase_visit_attributes);

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
