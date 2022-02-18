// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/op/multiclass_nms.hpp"

#include <cstring>
#include <ngraph/validation_util.hpp>

#include "itt.hpp"
#include "ngraph/attribute_visitor.hpp"
#include "ngraph/op/constant.hpp"
#include "ngraph/op/util/op_types.hpp"
#include "ngraph/runtime/reference/multiclass_nms.hpp"
#include "ngraph/type/bfloat16.hpp"
#include "ngraph/type/float16.hpp"
#include "ngraph/util.hpp"

using namespace ngraph;

BWDCMP_RTTI_DEFINITION(ov::op::v8::MulticlassNms);

op::v8::MulticlassNms::MulticlassNms() : NmsBase(m_attrs.output_type, m_attrs.nms_top_k, m_attrs.keep_top_k) {}

op::v8::MulticlassNms::MulticlassNms(const Output<Node>& boxes, const Output<Node>& scores, const Attributes& attrs)
    : NmsBase(boxes, scores, m_attrs.output_type, m_attrs.nms_top_k, m_attrs.keep_top_k),
      m_attrs{attrs} {
    constructor_validate_and_infer_types();
}

std::shared_ptr<Node> op::v8::MulticlassNms::clone_with_new_inputs(const OutputVector& new_args) const {
    NGRAPH_OP_SCOPE(v8_MulticlassNms_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    NODE_VALIDATION_CHECK(this, new_args.size() == 2, "Number of inputs must be 2");

    return std::make_shared<op::v8::MulticlassNms>(new_args.at(0), new_args.at(1), m_attrs);
}

void op::v8::MulticlassNms::validate() {
    NGRAPH_OP_SCOPE(v8_MulticlassNms_validate);
    NmsBase::validate();

    NODE_VALIDATION_CHECK(this,
                          m_attrs.background_class >= -1,
                          "The 'background_class' must be great or equal -1. Got:",
                          m_attrs.background_class);

    NODE_VALIDATION_CHECK(this,
                          m_attrs.nms_eta >= 0.0f && m_attrs.nms_eta <= 1.0f,
                          "The 'nms_eta' must be in close range [0, 1.0]. Got:",
                          m_attrs.nms_eta);
}

bool ngraph::op::v8::MulticlassNms::visit_attributes(AttributeVisitor& visitor) {
    NGRAPH_OP_SCOPE(v8_MulticlassNms_visit_attributes);

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
