// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <ngraph/opsets/opset3.hpp>
#include <ngraph/opsets/opset5.hpp>
#include "vpu/ngraph/operations/static_shape_non_maximum_suppression.hpp"

#include "ngraph/runtime/host_tensor.hpp"

namespace ngraph { namespace vpu { namespace op {

constexpr NodeTypeInfo StaticShapeNonMaxSuppression::type_info;

StaticShapeNonMaxSuppression::StaticShapeNonMaxSuppression(
        const Output<Node>& boxes,
        const Output<Node>& scores,
        const Output<Node>& maxOutputBoxesPerClass,
        const Output<Node>& iouThreshold,
        const Output<Node>& scoreThreshold,
        const Output<Node>& softNmsSigma,
        int centerPointBox,
        const bool sortResultDescending,
        const element::Type& outputType)
        : ngraph::op::NonMaxSuppressionIE3(
        boxes, scores, maxOutputBoxesPerClass, iouThreshold, scoreThreshold,
        softNmsSigma, centerPointBox, sortResultDescending, outputType) {
    constructor_validate_and_infer_types();
}

std::shared_ptr<Node>
StaticShapeNonMaxSuppression::clone_with_new_inputs(const OutputVector& new_args) const {
    check_new_args_count(this, new_args);
    return std::make_shared<StaticShapeNonMaxSuppression>(new_args.at(0), new_args.at(1), new_args.at(2), new_args.at(3),
                                                          new_args.at(4), new_args.at(5), m_center_point_box, m_sort_result_descending,
                                                          m_output_type);
}

void StaticShapeNonMaxSuppression::validate_and_infer_types() {
    ngraph::op::NonMaxSuppressionIE3::validate_and_infer_types();

    auto outIndicesShape = get_output_partial_shape(0);
    auto outScoresShape = get_output_partial_shape(1);

    NODE_VALIDATION_CHECK(this, outIndicesShape.is_static(),
                          "StaticShapeNonMaxSuppression output shape is not fully defined: ", outIndicesShape);
    NODE_VALIDATION_CHECK(this, outScoresShape.is_static(),
                          "StaticShapeNonMaxSuppression output shape is not fully defined: ", outScoresShape);

    // Replace valid outputs with the shape of selected_indices and selected_scores outputs
    set_output_type(2, m_output_type, Shape{2});
}

}  // namespace op
}  // namespace vpu
} // namespace ngraph
