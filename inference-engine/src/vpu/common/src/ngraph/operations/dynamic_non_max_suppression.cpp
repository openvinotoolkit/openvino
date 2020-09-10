// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpu/ngraph/operations/dynamic_non_max_suppression.hpp"

#include "vpu/utils/error.hpp"

#include "ngraph/opsets/opset3.hpp"
#include "ngraph/evaluator.hpp"

namespace ngraph { namespace vpu { namespace op {

constexpr NodeTypeInfo DynamicNonMaxSuppression::type_info;

DynamicNonMaxSuppression::DynamicNonMaxSuppression(
    const Output<Node>& boxes,
    const Output<Node>& scores,
    const Output<Node>& max_output_boxes_per_class,
    const Output<Node>& iou_threshold,
    const Output<Node>& score_threshold,
    const BoxEncodingType box_encoding,
    const bool sort_result_descending,
    const element::Type& output_type)
    : ngraph::op::v4::NonMaxSuppression(boxes,
                                        scores,
                                        max_output_boxes_per_class,
                                        iou_threshold,
                                        score_threshold,
                                        box_encoding,
                                        sort_result_descending,
                                        output_type) {
    constructor_validate_and_infer_types();
}

DynamicNonMaxSuppression::DynamicNonMaxSuppression(
    const Output<Node>& boxes,
    const Output<Node>& scores,
    const BoxEncodingType box_encoding,
    const bool sort_result_descending,
    const element::Type& output_type)
    : ngraph::op::v4::NonMaxSuppression(boxes,
                                        scores,
                                        ngraph::op::Constant::create(element::i64, Shape{}, {0}),
                                        ngraph::op::Constant::create(element::f32, Shape{}, {.0f}),
                                        ngraph::op::Constant::create(element::f32, Shape{}, {.0f}),
                                        box_encoding,
                                        sort_result_descending,
                                        output_type) {
    constructor_validate_and_infer_types();
}

std::shared_ptr<Node> DynamicNonMaxSuppression::clone_with_new_inputs(const OutputVector& new_args) const {
    check_new_args_count(this, new_args);
    NODE_VALIDATION_CHECK(this,
                          new_args.size() >= 2 && new_args.size() <= 5,
                          "Number of inputs must be 2, 3, 4 or 5");

    const auto& arg2 = new_args.size() > 2
                       ? new_args.at(2)
                       : ngraph::op::Constant::create(element::i32, Shape{}, {0});
    const auto& arg3 = new_args.size() > 3
                       ? new_args.at(3)
                       : ngraph::op::Constant::create(element::f32, Shape{}, {.0f});
    const auto& arg4 = new_args.size() > 4
                       ? new_args.at(4)
                       : ngraph::op::Constant::create(element::f32, Shape{}, {.0f});

    return std::make_shared<DynamicNonMaxSuppression>(new_args.at(0),
                                                      new_args.at(1),
                                                      arg2,
                                                      arg3,
                                                      arg4,
                                                      m_box_encoding,
                                                      m_sort_result_descending,
                                                      m_output_type);
}

void DynamicNonMaxSuppression::validate_and_infer_types() {
    ngraph::op::v4::NonMaxSuppression::validate_and_infer_types();

    // NonMaxSuppression produces triplets
    // that have the following format: [batch_index, class_index, box_index]
    set_output_type(0, m_output_type, PartialShape{Dimension::dynamic(), 3});
}

}  // namespace op
}  // namespace vpu
}  // namespace ngraph
