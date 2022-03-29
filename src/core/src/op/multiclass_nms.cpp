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
BWDCMP_RTTI_DEFINITION(ov::op::v9::MulticlassNms);

op::v8::MulticlassNms::MulticlassNms(const Output<Node>& boxes, const Output<Node>& scores, const Attributes& attrs)
    : MulticlassNmsBase({boxes, scores}, attrs) {
    constructor_validate_and_infer_types();
}

std::shared_ptr<Node> op::v8::MulticlassNms::clone_with_new_inputs(const OutputVector& new_args) const {
    NGRAPH_OP_SCOPE(MulticlassNms_v8_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    NODE_VALIDATION_CHECK(this, new_args.size() >= 2, "Number of inputs must be 2 at least");

    return std::make_shared<MulticlassNms>(new_args.at(0), new_args.at(1), m_attrs);
}

op::v9::MulticlassNms::MulticlassNms(const Output<Node>& boxes, const Output<Node>& scores, const Attributes& attrs)
    : MulticlassNmsBase({boxes, scores}, attrs) {
    constructor_validate_and_infer_types();
}

op::v9::MulticlassNms::MulticlassNms(const Output<Node>& boxes,
                                     const Output<Node>& scores,
                                     const Output<Node>& roisnum,
                                     const Attributes& attrs)
    : MulticlassNmsBase({boxes, scores, roisnum}, attrs) {
    constructor_validate_and_infer_types();
}

std::shared_ptr<Node> op::v9::MulticlassNms::clone_with_new_inputs(const OutputVector& new_args) const {
    NGRAPH_OP_SCOPE(MulticlassNms_v9_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    NODE_VALIDATION_CHECK(this, new_args.size() == 2 || new_args.size() == 3, "Number of inputs must be 2 or 3");

    if (new_args.size() == 3) {
        return std::make_shared<MulticlassNms>(new_args.at(0), new_args.at(1), new_args.at(2), m_attrs);
    } else if (new_args.size() == 2) {
        return std::make_shared<MulticlassNms>(new_args.at(0), new_args.at(1), m_attrs);
    }
    throw ngraph::ngraph_error("Unsupported number of inputs: " + std::to_string(new_args.size()));
}

namespace {
inline bool is_float_type_admissible(const ov::element::Type& t) {
    return t == ov::element::f32 || t == ov::element::f16 || t == ov::element::bf16;
}
}  // namespace

bool op::v9::MulticlassNms::validate() {
    NGRAPH_OP_SCOPE(MulticlassNms_v9_validate);

    const auto boxes_ps = get_input_partial_shape(0);
    const auto scores_ps = get_input_partial_shape(1);

    // validate dtype of each input
    NODE_VALIDATION_CHECK(this,
                          m_output_type == element::i64 || m_output_type == element::i32,
                          "Output type must be i32 or i64");

    NODE_VALIDATION_CHECK(this,
                          is_float_type_admissible(get_input_element_type(0)),
                          "Expected bf16, fp16 or fp32 as element type for the 'boxes' input.");

    NODE_VALIDATION_CHECK(this,
                          is_float_type_admissible(get_input_element_type(1)),
                          "Expected bf16, fp16 or fp32 as element type for the 'scores' input.");

    NODE_VALIDATION_CHECK(this,
                          get_input_element_type(0).compatible(get_input_element_type(1)),
                          "Expected 'boxes', 'scores' type is same.");

    // validate rank of each input
    if (boxes_ps.rank().is_dynamic() || scores_ps.rank().is_dynamic()) {
        return false;
    }

    if (get_input_size() == 3) {
        const auto roisnum_ps = get_input_partial_shape(2);
        if (roisnum_ps.rank().is_dynamic()) {
            return false;
        }
    }

    // validate shape of each input
    NODE_VALIDATION_CHECK(this,
                          boxes_ps.rank().is_static() && boxes_ps.rank().get_length() == 3,
                          "Expected a 3D tensor for the 'boxes' input. Got: ",
                          boxes_ps);

    NODE_VALIDATION_CHECK(this,
                          boxes_ps[2].is_static() && boxes_ps[2].get_length() == 4,
                          "The third dimension of the 'boxes' must be 4. Got: ",
                          boxes_ps[2]);

    NODE_VALIDATION_CHECK(
        this,
        scores_ps.rank().is_static() && (scores_ps.rank().get_length() == 3 || scores_ps.rank().get_length() == 2),
        "Expected a 2D or 3D tensor for the 'scores' input. Got: ",
        scores_ps);

    if (get_input_size() == 3) {
        const auto roisnum_ps = get_input_partial_shape(2);
        NODE_VALIDATION_CHECK(this,
                              roisnum_ps.rank().is_static() && roisnum_ps.rank().get_length() == 1,
                              "Expected a 1D tensor for the 'roisnum' input. Got: ",
                              roisnum_ps);
    }

    // validate attributes
    NODE_VALIDATION_CHECK(this, m_nms_top_k >= -1, "The 'nms_top_k' must be great or equal -1. Got:", m_nms_top_k);

    NODE_VALIDATION_CHECK(this, m_keep_top_k >= -1, "The 'keep_top_k' must be great or equal -1. Got:", m_keep_top_k);

    // validate compatibility of input shapes
    if (scores_ps.rank().is_static() && scores_ps.rank().get_length() == 3) {  // if scores shape (N, C, M)
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
    }

    if (scores_ps.rank().is_static() && scores_ps.rank().get_length() == 2) {  // if scores shape (C, M)
        const auto num_classes_boxes = boxes_ps[0];
        const auto num_classes_scores = scores_ps[0];
        NODE_VALIDATION_CHECK(this,
                              num_classes_boxes.same_scheme(num_classes_scores),
                              "'boxes' and 'scores' input shapes must match. Boxes: ",
                              num_classes_boxes,
                              "; Scores: ",
                              num_classes_scores);

        const auto num_boxes_boxes = boxes_ps[1];
        const auto num_boxes_scores = scores_ps[1];
        NODE_VALIDATION_CHECK(this,
                              num_boxes_boxes.same_scheme(num_boxes_scores),
                              "'boxes' and 'scores' input shapes must match. Boxes: ",
                              num_boxes_boxes,
                              "; Scores: ",
                              num_boxes_scores);

        NODE_VALIDATION_CHECK(this,
                              get_input_size() == 3,
                              "Expected the 'roisnum' input when the input 'scores' is a 2D tensor.");
    }

    return true;
}

void op::v9::MulticlassNms::validate_and_infer_types() {
    NGRAPH_OP_SCOPE(MulticlassNms_v9_validate_and_infer_types);

    //
    const auto boxes_ps = get_input_partial_shape(0);
    const auto scores_ps = get_input_partial_shape(1);

    // Here output 0 and output 1 is not the real dimension of output.
    // It will be rewritten in the computing runtime.
    // But we still need it here for static shape only backends.
    auto first_dim_shape = Dimension::dynamic();

    const auto validated = validate();

    if (validated) {  //  rank of inputs now are static, but dims maybe not.
        const bool shared = (scores_ps.rank().get_length() == 3);
        ov::PartialShape roisnum_ps;
        if (!shared) {
            roisnum_ps = get_input_partial_shape(2);
        }

        if ((shared && boxes_ps[1].is_static() && scores_ps[1].is_static() && scores_ps[0].is_static()) ||
            (!shared && boxes_ps[1].is_static() && boxes_ps[0].is_static() && roisnum_ps[0].is_static())) {
            const auto num_boxes = shared ? boxes_ps[1].get_length() : boxes_ps[1].get_length();
            const auto num_classes = shared ? scores_ps[1].get_length() : boxes_ps[0].get_length();
            auto num_images = shared ? scores_ps[0].get_length() : roisnum_ps[0].get_length();

            int64_t max_output_boxes_per_class = 0;
            if (m_nms_top_k >= 0)
                max_output_boxes_per_class = std::min(num_boxes, (int64_t)m_nms_top_k);
            else
                max_output_boxes_per_class = num_boxes;

            auto max_output_boxes_per_batch = max_output_boxes_per_class * num_classes;
            if (m_keep_top_k >= 0)
                max_output_boxes_per_batch = std::min(max_output_boxes_per_batch, (int64_t)m_keep_top_k);

            first_dim_shape = Dimension(0, max_output_boxes_per_batch * num_images);
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
    if (get_input_size() == 3) {  // !shared FIXME: tensor input && 3 inputs, possible?
        const auto roisnum_ps = get_input_partial_shape(2);
        if (roisnum_ps.rank().is_static() && roisnum_ps.rank().get_length() > 0) {
            set_output_type(2, m_output_type, {roisnum_ps[0]});
        } else {
            set_output_type(2, m_output_type, {Dimension::dynamic()});
        }
    } else {  // shared
        if (boxes_ps.rank().is_static() && boxes_ps.rank().get_length() > 0) {
            set_output_type(2, m_output_type, {boxes_ps[0]});
        } else {
            set_output_type(2, m_output_type, {Dimension::dynamic()});
        }
    }
}
