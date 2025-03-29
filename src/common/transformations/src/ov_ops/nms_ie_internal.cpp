// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ov_ops/nms_ie_internal.hpp"

#include <memory>

#include "itt.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/util/op_types.hpp"

using namespace std;
using namespace ov;

op::internal::NonMaxSuppressionIEInternal::NonMaxSuppressionIEInternal(const Output<Node>& boxes,
                                                                       const Output<Node>& scores,
                                                                       const Output<Node>& max_output_boxes_per_class,
                                                                       const Output<Node>& iou_threshold,
                                                                       const Output<Node>& score_threshold,
                                                                       int center_point_box,
                                                                       bool sort_result_descending,
                                                                       const ov::element::Type& output_type,
                                                                       const ov::element::Type& score_output_type,
                                                                       const int rotation)
    : Op({boxes, scores, max_output_boxes_per_class, iou_threshold, score_threshold}),
      m_center_point_box(center_point_box),
      m_sort_result_descending(sort_result_descending),
      m_output_type(output_type),
      m_scores_output_type(score_output_type),
      m_rotation(rotation) {
    constructor_validate_and_infer_types();
}

op::internal::NonMaxSuppressionIEInternal::NonMaxSuppressionIEInternal(const Output<Node>& boxes,
                                                                       const Output<Node>& scores,
                                                                       const Output<Node>& max_output_boxes_per_class,
                                                                       const Output<Node>& iou_threshold,
                                                                       const Output<Node>& score_threshold,
                                                                       const Output<Node>& soft_nms_sigma,
                                                                       int center_point_box,
                                                                       bool sort_result_descending,
                                                                       const ov::element::Type& output_type,
                                                                       const ov::element::Type& score_output_type,
                                                                       const int rotation)
    : Op({boxes, scores, max_output_boxes_per_class, iou_threshold, score_threshold, soft_nms_sigma}),
      m_center_point_box(center_point_box),
      m_sort_result_descending(sort_result_descending),
      m_output_type(output_type),
      m_scores_output_type(score_output_type),
      m_rotation{rotation} {
    constructor_validate_and_infer_types();
}

std::shared_ptr<Node> op::internal::NonMaxSuppressionIEInternal::clone_with_new_inputs(
    const ov::OutputVector& new_args) const {
    INTERNAL_OP_SCOPE(internal_NonMaxSuppressionIEInternal_clone_with_new_inputs);
    if (new_args.size() == 6) {
        return make_shared<NonMaxSuppressionIEInternal>(new_args.at(0),
                                                        new_args.at(1),
                                                        new_args.at(2),
                                                        new_args.at(3),
                                                        new_args.at(4),
                                                        new_args.at(5),
                                                        m_center_point_box,
                                                        m_sort_result_descending,
                                                        m_output_type,
                                                        m_scores_output_type,
                                                        m_rotation);
    } else if (new_args.size() == 5) {
        return make_shared<NonMaxSuppressionIEInternal>(new_args.at(0),
                                                        new_args.at(1),
                                                        new_args.at(2),
                                                        new_args.at(3),
                                                        new_args.at(4),
                                                        m_center_point_box,
                                                        m_sort_result_descending,
                                                        m_output_type,
                                                        m_scores_output_type,
                                                        m_rotation);
    }
    OPENVINO_THROW("Unsupported number of inputs: " + std::to_string(new_args.size()));
}

bool op::internal::NonMaxSuppressionIEInternal::visit_attributes(AttributeVisitor& visitor) {
    INTERNAL_OP_SCOPE(internal_NonMaxSuppressionIEInternal_visit_attributes);
    visitor.on_attribute("center_point_box", m_center_point_box);
    visitor.on_attribute("sort_result_descending", m_sort_result_descending);
    visitor.on_attribute("output_type", m_output_type);
    visitor.on_attribute("score_output_type", m_scores_output_type);
    visitor.on_attribute("rotation", m_rotation);
    return true;
}

static constexpr size_t boxes_port = 0;
static constexpr size_t scores_port = 1;
static constexpr size_t max_output_boxes_per_class_port = 2;

int64_t op::internal::NonMaxSuppressionIEInternal::max_boxes_output_from_input() const {
    int64_t max_output_boxes{0};

    size_t num_of_inputs = inputs().size();
    if (num_of_inputs < 3) {
        return 0;
    }

    const auto max_output_boxes_input =
        ov::as_type_ptr<const ov::op::v0::Constant>(input_value(max_output_boxes_per_class_port).get_node_shared_ptr());
    max_output_boxes = max_output_boxes_input->cast_vector<int64_t>().at(0);

    return max_output_boxes;
}

void op::internal::NonMaxSuppressionIEInternal::validate_and_infer_types() {
    INTERNAL_OP_SCOPE(internal_NonMaxSuppressionIEInternal_validate_and_infer_types);
    const auto boxes_ps = get_input_partial_shape(boxes_port);
    const auto scores_ps = get_input_partial_shape(scores_port);

    // NonMaxSuppression produces triplets
    // that have the following format: [batch_index, class_index, box_index]
    PartialShape out_shape = {Dimension::dynamic(), 3};

    if (boxes_ps.rank().is_static() && scores_ps.rank().is_static()) {
        const auto num_boxes_boxes = boxes_ps[1];
        const auto max_output_boxes_per_class_node = input_value(max_output_boxes_per_class_port).get_node_shared_ptr();
        if (num_boxes_boxes.is_static() && scores_ps[0].is_static() && scores_ps[1].is_static() &&
            op::util::is_constant(max_output_boxes_per_class_node)) {
            const auto num_boxes = num_boxes_boxes.get_length();
            const auto num_classes = scores_ps[1].get_length();
            const auto max_output_boxes_per_class = max_boxes_output_from_input();

            out_shape[0] = std::min(num_boxes, max_output_boxes_per_class) * num_classes * scores_ps[0].get_length();
        }
    }

    set_output_type(0, m_output_type, out_shape);
    set_output_type(1, m_scores_output_type, out_shape);
    set_output_type(2, m_output_type, Shape{1});
}
