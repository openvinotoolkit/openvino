// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/op/util/nms_base.hpp"

#include <cstring>
#include <ngraph/validation_util.hpp>

#include "itt.hpp"
#include "ngraph/attribute_visitor.hpp"
#include "ngraph/op/constant.hpp"
#include "ngraph/op/util/op_types.hpp"
#include "ngraph/type/bfloat16.hpp"
#include "ngraph/type/float16.hpp"
#include "ngraph/util.hpp"

ov::op::util::NmsBase::NmsBase(ngraph::element::Type& output_type, int& nms_top_k, int& keep_top_k)
    : m_output_type(output_type),
      m_nms_top_k(nms_top_k),
      m_keep_top_k(keep_top_k) {}

ov::op::util::NmsBase::NmsBase(const OutputVector& arguments,
                               ngraph::element::Type& output_type,
                               int& nms_top_k,
                               int& keep_top_k)
    : Op(arguments),
      m_output_type(output_type),
      m_nms_top_k(nms_top_k),
      m_keep_top_k(keep_top_k) {}

namespace {
inline bool is_float_type_admissible(const ov::element::Type& t) {
    return t == ov::element::f32 || t == ov::element::f16 || t == ov::element::bf16;
}
}  // namespace

bool ov::op::util::NmsBase::validate() {
    NGRAPH_OP_SCOPE(util_NmsBase_validate);

    const auto boxes_ps = get_input_partial_shape(0);
    const auto scores_ps = get_input_partial_shape(1);

    NODE_VALIDATION_CHECK(this,
                          m_output_type == element::i64 || m_output_type == element::i32,
                          "Output type must be i32 or i64");

    if (boxes_ps.is_dynamic() || scores_ps.is_dynamic()) {
        return false;
    }

    NODE_VALIDATION_CHECK(this,
                          is_float_type_admissible(get_input_element_type(0)),
                          "Expected bf16, fp16 or fp32 as element type for the 'boxes' input.");

    NODE_VALIDATION_CHECK(this,
                          is_float_type_admissible(get_input_element_type(1)),
                          "Expected bf16, fp16 or fp32 as element type for the 'scores' input.");

    NODE_VALIDATION_CHECK(this,
                          get_input_element_type(0).compatible(get_input_element_type(1)),
                          "Expected 'boxes', 'scores' type is same.");

    NODE_VALIDATION_CHECK(this,
                          boxes_ps.rank().is_static() && boxes_ps.rank().get_length() == 3,
                          "Expected a 3D tensor for the 'boxes' input. Got: ",
                          boxes_ps);

    NODE_VALIDATION_CHECK(this,
                          boxes_ps[2].is_static() && boxes_ps[2].get_length() == 4,
                          "The third dimension of the 'boxes' must be 4. Got: ",
                          boxes_ps[2]);

    NODE_VALIDATION_CHECK(this,
                          scores_ps.rank().is_static() && scores_ps.rank().get_length() == 3,
                          "Expected a 3D tensor for the 'scores' input. Got: ",
                          scores_ps);

    NODE_VALIDATION_CHECK(this, m_nms_top_k >= -1, "The 'nms_top_k' must be great or equal -1. Got:", m_nms_top_k);

    NODE_VALIDATION_CHECK(this, m_keep_top_k >= -1, "The 'keep_top_k' must be great or equal -1. Got:", m_keep_top_k);

    const auto num_batches_boxes = boxes_ps[0];
    const auto num_batches_scores = scores_ps[0];

    NODE_VALIDATION_CHECK(this,
                          num_batches_boxes.same_scheme(num_batches_scores),
                          "The first dimension of both 'boxes' and 'scores' must match. Boxes: ",
                          num_batches_boxes,
                          "; Scores: ",
                          num_batches_scores);

    const auto num_boxes_boxes = boxes_ps[1];
    const auto num_boxes_scores = scores_ps[2];
    NODE_VALIDATION_CHECK(this,
                          num_boxes_boxes.same_scheme(num_boxes_scores),
                          "'boxes' and 'scores' input shapes must match at the second and third "
                          "dimension respectively. Boxes: ",
                          num_boxes_boxes,
                          "; Scores: ",
                          num_boxes_scores);

    return true;
}

void ov::op::util::NmsBase::validate_and_infer_types() {
    NGRAPH_OP_SCOPE(util_NmsBase_validate_and_infer_types);
    const auto boxes_ps = get_input_partial_shape(0);
    const auto scores_ps = get_input_partial_shape(1);

    auto first_dim_shape = Dimension::dynamic();

    validate();

    if (boxes_ps.rank().is_static() && scores_ps.rank().is_static()) {
        const auto num_boxes_boxes = boxes_ps[1];
        if (num_boxes_boxes.is_static() && scores_ps[0].is_static() && scores_ps[1].is_static()) {
            const auto num_boxes = num_boxes_boxes.get_length();
            const auto num_classes = scores_ps[1].get_length();
            int64_t max_output_boxes_per_class = 0;
            if (m_nms_top_k >= 0)
                max_output_boxes_per_class = std::min(num_boxes, (int64_t)m_nms_top_k);
            else
                max_output_boxes_per_class = num_boxes;

            auto max_output_boxes_per_batch = max_output_boxes_per_class * num_classes;
            if (m_keep_top_k >= 0)
                max_output_boxes_per_batch = std::min(max_output_boxes_per_batch, (int64_t)m_keep_top_k);

            first_dim_shape = Dimension(0, max_output_boxes_per_batch * scores_ps[0].get_length());
        }
    }

    // 'selected_outputs' have the following format:
    //      [number of selected boxes, [class_id, box_score, xmin, ymin, xmax, ymax]]
    set_output_type(0, get_input_element_type(0), {first_dim_shape, 6});
    // 'selected_indices' have the following format:
    //      [number of selected boxes, ]
    set_output_type(1, m_output_type, {first_dim_shape, 1});
    // 'selected_num' have the following format:
    //      [num_batches, ]
    if (boxes_ps.rank().is_static() && boxes_ps.rank().get_length() > 0) {
        set_output_type(2, m_output_type, {boxes_ps[0]});
    } else {
        set_output_type(2, m_output_type, {Dimension::dynamic()});
    }
}

std::ostream& ov::operator<<(std::ostream& s, const op::util::NmsBase::SortResultType& type) {
    return s << as_string(type);
}

namespace ov {
template <>
NGRAPH_API EnumNames<op::util::NmsBase::SortResultType>& EnumNames<op::util::NmsBase::SortResultType>::get() {
    static auto enum_names =
        EnumNames<op::util::NmsBase::SortResultType>("op::util::NmsBase::SortResultType",
                                                     {{"classid", op::util::NmsBase::SortResultType::CLASSID},
                                                      {"score", op::util::NmsBase::SortResultType::SCORE},
                                                      {"none", op::util::NmsBase::SortResultType::NONE}});
    return enum_names;
}

BWDCMP_RTTI_DEFINITION(AttributeAdapter<op::util::NmsBase::SortResultType>);
}  // namespace ov
