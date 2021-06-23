// Copyright (C) 2018-2021 Intel Corporation
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

NGRAPH_RTTI_DEFINITION(op::v8::MulticlassNms, "MulticlassNms", 8);

op::v8::MulticlassNms::MulticlassNms(const Output<Node>& boxes,
                                     const Output<Node>& scores,
                                     const SortResultType sort_result_type,
                                     bool sort_result_across_batch,
                                     const ngraph::element::Type& output_type,
                                     const float iou_threshold,
                                     const float score_threshold,
                                     const int nms_top_k,
                                     const int keep_top_k,
                                     const int background_class,
                                     const float nms_eta,
                                     const bool normalized)
    : NmsBase(boxes, scores, sort_result_type, output_type, nms_top_k, keep_top_k)
    , m_sort_result_across_batch{sort_result_across_batch}
    , m_iou_threshold{iou_threshold}
    , m_score_threshold{score_threshold}
    , m_background_class{background_class}
    , m_nms_eta{nms_eta}
    , m_normalized{normalized}
{
    constructor_validate_and_infer_types();
}

std::shared_ptr<Node>
    op::v8::MulticlassNms::clone_with_new_inputs(const OutputVector& new_args) const
{
    NGRAPH_OP_SCOPE(v8_MulticlassNms_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    NODE_VALIDATION_CHECK(this, new_args.size() == 2, "Number of inputs must be 2");

    return std::make_shared<op::v8::MulticlassNms>(new_args.at(0),
                                                   new_args.at(1),
                                                   m_sort_result_type,
                                                   m_sort_result_across_batch,
                                                   m_output_type,
                                                   m_iou_threshold,
                                                   m_score_threshold,
                                                   m_nms_top_k,
                                                   m_keep_top_k,
                                                   m_background_class,
                                                   m_nms_eta,
                                                   m_normalized);
}

void op::v8::MulticlassNms::validate()
{
    NmsBase::validate();

    NODE_VALIDATION_CHECK(this,
                          m_background_class >= -1,
                          "The 'background_class' must be great or equal -1. Got:",
                          m_background_class);

    NODE_VALIDATION_CHECK(this,
                          m_nms_eta >= 0.0f && m_nms_eta <= 1.0f,
                          "The 'nms_eta' must be in close range [0, 1.0]. Got:",
                          m_nms_eta);
}

bool ngraph::op::v8::MulticlassNms::visit_attributes(AttributeVisitor& visitor)
{
    NGRAPH_OP_SCOPE(v8_MulticlassNms_visit_attributes);
    NmsBase::visit_attributes(visitor);

    visitor.on_attribute("sort_result_across_batch", m_sort_result_across_batch);
    visitor.on_attribute("iou_threshold", m_iou_threshold);
    visitor.on_attribute("score_threshold", m_score_threshold);
    visitor.on_attribute("background_class", m_background_class);
    visitor.on_attribute("nms_eta", m_nms_eta);
    visitor.on_attribute("normalized", m_normalized);

    return true;
}
