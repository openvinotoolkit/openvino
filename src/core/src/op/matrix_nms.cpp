// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/op/matrix_nms.hpp"

#include <cstring>
#include <ngraph/validation_util.hpp>

#include "itt.hpp"
#include "ngraph/attribute_visitor.hpp"
#include "ngraph/op/constant.hpp"
#include "ngraph/op/util/op_types.hpp"
#include "ngraph/runtime/reference/matrix_nms.hpp"
#include "ngraph/type/bfloat16.hpp"
#include "ngraph/type/float16.hpp"
#include "ngraph/util.hpp"

using namespace ngraph;
BWDCMP_RTTI_DEFINITION(op::v8::MatrixNms);

op::v8::MatrixNms::MatrixNms(const Output<Node>& boxes, const Output<Node>& scores, const Attributes& attrs)
    : Op({boxes, scores}),
      m_attrs{attrs} {
    constructor_validate_and_infer_types();
}

std::shared_ptr<Node> op::v8::MatrixNms::clone_with_new_inputs(const OutputVector& new_args) const {
    OV_OP_SCOPE(v8_MatrixNms_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    NODE_VALIDATION_CHECK(this, new_args.size() == 2, "Number of inputs must be 2");

    return std::make_shared<op::v8::MatrixNms>(new_args.at(0), new_args.at(1), m_attrs);
}

void op::v8::MatrixNms::validate() {
    OV_OP_SCOPE(v8_MatrixNms_validate);

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

    // validate attributes
    NODE_VALIDATION_CHECK(this, nms_top_k >= -1, "The 'nms_top_k' must be great or equal -1. Got:", nms_top_k);

    NODE_VALIDATION_CHECK(this, keep_top_k >= -1, "The 'keep_top_k' must be great or equal -1. Got:", keep_top_k);

    NODE_VALIDATION_CHECK(this,
                          m_attrs.background_class >= -1,
                          "The 'background_class' must be great or equal -1. Got:",
                          m_attrs.background_class);
}

void op::v8::MatrixNms::validate_and_infer_types() {
    OV_OP_SCOPE(v8_MatrixNms_validate_and_infer_types);
    const auto boxes_ps = get_input_partial_shape(0);
    const auto scores_ps = get_input_partial_shape(1);

    auto first_dim_shape = Dimension::dynamic();

    validate();

    const auto& nms_attrs = this->get_attrs();
    const auto output_type = nms_attrs.output_type;
    const auto nms_top_k = nms_attrs.nms_top_k;
    const auto keep_top_k = nms_attrs.keep_top_k;

    if (boxes_ps.rank().is_static() && scores_ps.rank().is_static()) {
        const auto num_boxes_boxes = boxes_ps[1];
        if (num_boxes_boxes.is_static() && scores_ps[0].is_static() && scores_ps[1].is_static()) {
            const auto num_boxes = num_boxes_boxes.get_length();
            const auto num_classes = scores_ps[1].get_length();
            int64_t max_output_boxes_per_class = 0;
            if (nms_top_k >= 0)
                max_output_boxes_per_class = std::min(num_boxes, (int64_t)nms_top_k);
            else
                max_output_boxes_per_class = num_boxes;

            auto max_output_boxes_per_batch = max_output_boxes_per_class * num_classes;
            if (keep_top_k >= 0)
                max_output_boxes_per_batch = std::min(max_output_boxes_per_batch, (int64_t)keep_top_k);

            first_dim_shape = Dimension(0, max_output_boxes_per_batch * scores_ps[0].get_length());
        }
    }

    // 'selected_outputs' have the following format:
    //      [number of selected boxes, [class_id, box_score, xmin, ymin, xmax, ymax]]
    set_output_type(0, get_input_element_type(0), {first_dim_shape, 6});
    // 'selected_indices' have the following format:
    //      [number of selected boxes, ]
    set_output_type(1, output_type, {first_dim_shape, 1});
    // 'selected_num' have the following format:
    //      [num_batches, ]
    if (boxes_ps.rank().is_static() && boxes_ps.rank().get_length() > 0) {
        set_output_type(2, output_type, {boxes_ps[0]});
    } else {
        set_output_type(2, output_type, {Dimension::dynamic()});
    }
}

bool op::v8::MatrixNms::visit_attributes(AttributeVisitor& visitor) {
    OV_OP_SCOPE(v8_MatrixNms_visit_attributes);

    visitor.on_attribute("sort_result_type", m_attrs.sort_result_type);
    visitor.on_attribute("output_type", m_attrs.output_type);
    visitor.on_attribute("nms_top_k", m_attrs.nms_top_k);
    visitor.on_attribute("keep_top_k", m_attrs.keep_top_k);
    visitor.on_attribute("sort_result_across_batch", m_attrs.sort_result_across_batch);
    visitor.on_attribute("score_threshold", m_attrs.score_threshold);
    visitor.on_attribute("background_class", m_attrs.background_class);
    visitor.on_attribute("decay_function", m_attrs.decay_function);
    visitor.on_attribute("gaussian_sigma", m_attrs.gaussian_sigma);
    visitor.on_attribute("post_threshold", m_attrs.post_threshold);
    visitor.on_attribute("normalized", m_attrs.normalized);

    return true;
}

std::ostream& ov::operator<<(std::ostream& s, const op::v8::MatrixNms::DecayFunction& type) {
    return s << as_string(type);
}

namespace ov {
template <>
NGRAPH_API EnumNames<ngraph::op::v8::MatrixNms::DecayFunction>&
EnumNames<ngraph::op::v8::MatrixNms::DecayFunction>::get() {
    static auto enum_names = EnumNames<ngraph::op::v8::MatrixNms::DecayFunction>(
        "op::v8::MatrixNms::DecayFunction",
        {{"gaussian", ngraph::op::v8::MatrixNms::DecayFunction::GAUSSIAN},
         {"linear", ngraph::op::v8::MatrixNms::DecayFunction::LINEAR}});
    return enum_names;
}

BWDCMP_RTTI_DEFINITION(AttributeAdapter<op::v8::MatrixNms::DecayFunction>);

}  // namespace ov

std::ostream& ov::operator<<(std::ostream& s, const op::v8::MatrixNms::SortResultType& type) {
    return s << as_string(type);
}

namespace ov {
template <>
NGRAPH_API EnumNames<op::v8::MatrixNms::SortResultType>& EnumNames<op::v8::MatrixNms::SortResultType>::get() {
    static auto enum_names =
        EnumNames<op::v8::MatrixNms::SortResultType>("op::v8::MatrixNms::SortResultType",
                                                     {{"classid", op::v8::MatrixNms::SortResultType::CLASSID},
                                                      {"score", op::v8::MatrixNms::SortResultType::SCORE},
                                                      {"none", op::v8::MatrixNms::SortResultType::NONE}});
    return enum_names;
}

BWDCMP_RTTI_DEFINITION(AttributeAdapter<op::v8::MatrixNms::SortResultType>);

}  // namespace ov
