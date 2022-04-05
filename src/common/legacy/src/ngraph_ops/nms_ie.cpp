// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "legacy/ngraph_ops/nms_ie.hpp"

#include <memory>

#include <ngraph/opsets/opset3.hpp>
#include <ngraph/opsets/opset4.hpp>

using namespace std;
using namespace ngraph;

BWDCMP_RTTI_DEFINITION(op::NonMaxSuppressionIE);
BWDCMP_RTTI_DEFINITION(op::NonMaxSuppressionIE2);
BWDCMP_RTTI_DEFINITION(op::NonMaxSuppressionIE3);

op::NonMaxSuppressionIE::NonMaxSuppressionIE(const Output<Node> &boxes,
                                             const Output<Node> &scores,
                                             const Output<Node> &max_output_boxes_per_class,
                                             const Output<Node> &iou_threshold,
                                             const Output<Node> &score_threshold,
                                             int center_point_box,
                                             bool sort_result_descending,
                                             const ngraph::element::Type& output_type)
        : Op({boxes, scores, max_output_boxes_per_class, iou_threshold, score_threshold}),
          m_center_point_box(center_point_box), m_sort_result_descending(sort_result_descending), m_output_type(output_type) {
    constructor_validate_and_infer_types();
}


std::shared_ptr<Node> op::NonMaxSuppressionIE::clone_with_new_inputs(const ngraph::OutputVector &new_args) const {
    check_new_args_count(this, new_args);
    return make_shared<NonMaxSuppressionIE>(new_args.at(0), new_args.at(1), new_args.at(2), new_args.at(3),
                                            new_args.at(4), m_center_point_box, m_sort_result_descending, m_output_type);
}

void op::NonMaxSuppressionIE::validate_and_infer_types() {
    auto squeeze_input = [](const Output<Node> &input) -> std::shared_ptr<Node> {
        return std::make_shared<opset3::Squeeze>(input, opset3::Constant::create(element::i64, Shape{1}, {0}));
    };

    // Calculate output shape using opset3::NonMaxSuppression
    auto max_output_boxes_per_class = std::dynamic_pointer_cast<opset3::Constant>(input_value(2).get_node_shared_ptr());
    auto nms = std::make_shared<opset3::NonMaxSuppression>(input_value(0), input_value(1),
            /* second input is used for output calculation and only if it's Constant output shape won't be dynamic */
                                                           max_output_boxes_per_class ? opset3::Constant::create(element::i64, Shape{},
                                                                   max_output_boxes_per_class->cast_vector<int64_t>()) : squeeze_input(input_value(2)),
                                                           squeeze_input(input_value(3)),
                                                           squeeze_input(input_value(4)),
                                                           opset3::NonMaxSuppression::BoxEncodingType::CENTER,
                                                           m_sort_result_descending,
                                                           m_output_type);
    set_output_type(0, nms->output(0).get_element_type(), nms->output(0).get_partial_shape());
}

bool op::NonMaxSuppressionIE::visit_attributes(AttributeVisitor& visitor) {
    visitor.on_attribute("center_point_box", m_center_point_box);
    visitor.on_attribute("sort_result_descending", m_sort_result_descending);
    visitor.on_attribute("output_type", m_output_type);
    return true;
}

// The second version of the operation is different just in the shape infer function (uses v4::NMS)
op::NonMaxSuppressionIE2::NonMaxSuppressionIE2(const Output<Node> &boxes,
                                               const Output<Node> &scores,
                                               const Output<Node> &max_output_boxes_per_class,
                                               const Output<Node> &iou_threshold,
                                               const Output<Node> &score_threshold,
                                               int center_point_box,
                                               bool sort_result_descending,
                                               const ngraph::element::Type& output_type)
        : op::NonMaxSuppressionIE(boxes, scores, max_output_boxes_per_class, iou_threshold, score_threshold, center_point_box, sort_result_descending,
                                  output_type) {
    constructor_validate_and_infer_types();
}


std::shared_ptr<Node> op::NonMaxSuppressionIE2::clone_with_new_inputs(const ngraph::OutputVector &new_args) const {
    check_new_args_count(this, new_args);
    return make_shared<NonMaxSuppressionIE2>(new_args.at(0), new_args.at(1), new_args.at(2), new_args.at(3),
                                             new_args.at(4), m_center_point_box, m_sort_result_descending, m_output_type);
}

void op::NonMaxSuppressionIE2::validate_and_infer_types() {
    auto squeeze_input = [](const Output<Node> &input) -> std::shared_ptr<Node> {
        return std::make_shared<opset4::Squeeze>(input, opset4::Constant::create(element::i64, Shape{1}, {0}));
    };

    // Calculate output shape using opset4::NonMaxSuppression
    auto max_output_boxes_per_class = std::dynamic_pointer_cast<opset4::Constant>(input_value(2).get_node_shared_ptr());
    auto nms = std::make_shared<opset4::NonMaxSuppression>(input_value(0), input_value(1),
            /* second input is used for output calculation and only if it's Constant output shape won't be dynamic */
                                                           max_output_boxes_per_class ? opset4::Constant::create(element::i64, Shape{},
                                                                   max_output_boxes_per_class->cast_vector<int64_t>()) : squeeze_input(input_value(2)),
                                                           squeeze_input(input_value(3)),
                                                           squeeze_input(input_value(4)),
                                                           opset4::NonMaxSuppression::BoxEncodingType::CENTER,
                                                           m_sort_result_descending,
                                                           m_output_type);
    set_output_type(0, nms->output(0).get_element_type(), nms->output(0).get_partial_shape());
}

op::NonMaxSuppressionIE3::NonMaxSuppressionIE3(const Output<Node>& boxes,
                                               const Output<Node>& scores,
                                               const Output<Node>& max_output_boxes_per_class,
                                               const Output<Node>& iou_threshold,
                                               const Output<Node>& score_threshold,
                                               int center_point_box,
                                               bool sort_result_descending,
                                               const ngraph::element::Type& output_type)
        : Op({boxes, scores, max_output_boxes_per_class, iou_threshold, score_threshold}),
          m_center_point_box(center_point_box), m_sort_result_descending(sort_result_descending), m_output_type(output_type) {
    constructor_validate_and_infer_types();
}

op::NonMaxSuppressionIE3::NonMaxSuppressionIE3(const Output<Node>& boxes,
                                               const Output<Node>& scores,
                                               const Output<Node>& max_output_boxes_per_class,
                                               const Output<Node>& iou_threshold,
                                               const Output<Node>& score_threshold,
                                               const Output<Node>& soft_nms_sigma,
                                               int center_point_box,
                                               bool sort_result_descending,
                                               const ngraph::element::Type& output_type)
        : Op({boxes, scores, max_output_boxes_per_class, iou_threshold, score_threshold, soft_nms_sigma}),
          m_center_point_box(center_point_box), m_sort_result_descending(sort_result_descending), m_output_type(output_type) {
    constructor_validate_and_infer_types();
}

std::shared_ptr<Node> op::NonMaxSuppressionIE3::clone_with_new_inputs(const ngraph::OutputVector &new_args) const {
    if (new_args.size() == 6) {
        return make_shared<NonMaxSuppressionIE3>(new_args.at(0), new_args.at(1), new_args.at(2), new_args.at(3),
                                             new_args.at(4), new_args.at(5), m_center_point_box, m_sort_result_descending,
                                             m_output_type);
    } else if (new_args.size() == 5) {
        return make_shared<NonMaxSuppressionIE3>(new_args.at(0), new_args.at(1), new_args.at(2), new_args.at(3),
                                             new_args.at(4), m_center_point_box, m_sort_result_descending,
                                             m_output_type);
    }
    throw ngraph::ngraph_error("Unsupported number of inputs: " + std::to_string(new_args.size()));
}

bool op::NonMaxSuppressionIE3::visit_attributes(AttributeVisitor& visitor) {
    visitor.on_attribute("center_point_box", m_center_point_box);
    visitor.on_attribute("sort_result_descending", m_sort_result_descending);
    visitor.on_attribute("output_type", m_output_type);
    return true;
}

static constexpr size_t boxes_port = 0;
static constexpr size_t scores_port = 1;
static constexpr size_t max_output_boxes_per_class_port = 2;

int64_t op::NonMaxSuppressionIE3::max_boxes_output_from_input() const {
    int64_t max_output_boxes{0};

    size_t num_of_inputs = inputs().size();
    if (num_of_inputs < 3) {
        return 0;
    }

    const auto max_output_boxes_input =
        ov::as_type_ptr<const op::Constant>(input_value(max_output_boxes_per_class_port).get_node_shared_ptr());
    max_output_boxes = max_output_boxes_input->cast_vector<int64_t>().at(0);

    return max_output_boxes;
}

void op::NonMaxSuppressionIE3::validate_and_infer_types() {
    const auto boxes_ps = get_input_partial_shape(boxes_port);
    const auto scores_ps = get_input_partial_shape(scores_port);

    // NonMaxSuppression produces triplets
    // that have the following format: [batch_index, class_index, box_index]
    PartialShape out_shape = {Dimension::dynamic(), 3};

    if (boxes_ps.rank().is_static() && scores_ps.rank().is_static()) {
        const auto num_boxes_boxes = boxes_ps[1];
        const auto max_output_boxes_per_class_node = input_value(max_output_boxes_per_class_port).get_node_shared_ptr();
        if (num_boxes_boxes.is_static() && scores_ps[0].is_static() && scores_ps[1].is_static() &&
            op::is_constant(max_output_boxes_per_class_node)) {
            const auto num_boxes = num_boxes_boxes.get_length();
            const auto num_classes = scores_ps[1].get_length();
            const auto max_output_boxes_per_class = max_boxes_output_from_input();

            out_shape[0] = std::min(num_boxes, max_output_boxes_per_class) * num_classes *
                           scores_ps[0].get_length();
        }
    }

    set_output_type(0, m_output_type, out_shape);
    set_output_type(1, element::f32, out_shape);
    set_output_type(2, m_output_type, Shape{1});
}
