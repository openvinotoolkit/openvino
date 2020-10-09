// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph_ops/nms_ie.hpp"

#include <memory>

#include <ngraph/opsets/opset3.hpp>
#include <ngraph/opsets/opset4.hpp>

using namespace std;
using namespace ngraph;

constexpr NodeTypeInfo op::NonMaxSuppressionIE::type_info;

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
constexpr NodeTypeInfo op::NonMaxSuppressionIE2::type_info;

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
