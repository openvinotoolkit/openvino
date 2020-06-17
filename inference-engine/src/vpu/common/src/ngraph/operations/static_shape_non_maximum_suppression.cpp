// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <ngraph/opsets/opset3.hpp>
#include "vpu/ngraph/operations/static_shape_non_maximum_suppression.hpp"

#include "ngraph/runtime/host_tensor.hpp"

namespace ngraph { namespace vpu { namespace op {

constexpr NodeTypeInfo StaticShapeNonMaxSuppression::type_info;

StaticShapeNonMaxSuppression::StaticShapeNonMaxSuppression(
        const Output<Node>& boxes,
        const Output<Node>& scores,
        const Output<Node>& max_output_boxes_per_class,
        const Output<Node>& iou_threshold,
        const Output<Node>& score_threshold,
        const StaticShapeNonMaxSuppression::BoxEncodingType box_encoding,
        const bool sort_result_descending,
        const element::Type& output_type)
        : ngraph::op::v3::NonMaxSuppression({
            boxes, scores, max_output_boxes_per_class, iou_threshold, score_threshold,
            box_encoding, sort_result_descending, output_type}) {
    constructor_validate_and_infer_types();
}

StaticShapeNonMaxSuppression::StaticShapeNonMaxSuppression(
        const Output<Node>& boxes,
        const Output<Node>& scores,
        const StaticShapeNonMaxSuppression::BoxEncodingType box_encoding,
        const bool sort_result_descending,
        const element::Type& output_type)
        : ngraph::op::v3::NonMaxSuppression({boxes,
              scores,
              ngraph::opset3::Constant::create(element::i64, Shape{}, {0}),
              ngraph::opset3::Constant::create(element::f32, Shape{}, {.0f}),
              ngraph::opset3::Constant::create(element::f32, Shape{}, {.0f}),
              box_encoding, sort_result_descending, output_type}) {
    constructor_validate_and_infer_types();
}

std::shared_ptr<Node>
StaticShapeNonMaxSuppression::clone_with_new_inputs(const OutputVector& new_args) const {
    check_new_args_count(this, new_args);
    NODE_VALIDATION_CHECK(this,
                          new_args.size() >= 2 && new_args.size() <= 5,
                          "Number of inputs must be 2, 3, 4 or 5");

    const auto& arg2 = new_args.size() > 2 ? new_args.at(2) : ngraph::opset3::Constant::create(element::i32, Shape{}, {0});
    const auto& arg3 = new_args.size() > 3 ? new_args.at(3) : ngraph::opset3::Constant::create(element::f32, Shape{}, {.0f});
    const auto& arg4 = new_args.size() > 4 ? new_args.at(4) : ngraph::opset3::Constant::create(element::f32, Shape{}, {.0f});

    return std::make_shared<StaticShapeNonMaxSuppression>(
          new_args.at(0),
          new_args.at(1),
          arg2,
          arg3,
          arg4,
          m_box_encoding,
          m_sort_result_descending,
          m_output_type);
}

void StaticShapeNonMaxSuppression::validate_and_infer_types() {
    ngraph::op::v3::NonMaxSuppression::validate_and_infer_types();

    const auto out_shape = this->get_output_partial_shape(0);
    NODE_VALIDATION_CHECK(this, out_shape.is_static(),
                          "StaticShapeTopK output shape is not fully defined: ", out_shape);

    set_output_size(2);
    set_output_type(0, m_output_type, out_shape);
    set_output_type(1, ngraph::element::i64, Shape{2});
}

}  // namespace op
}  // namespace vpu
} // namespace ngraph
