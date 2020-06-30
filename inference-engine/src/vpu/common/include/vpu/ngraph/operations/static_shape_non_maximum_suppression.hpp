// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ngraph/node.hpp>
#include <ngraph/op/op.hpp>

#include <memory>
#include <vector>

namespace ngraph { namespace vpu { namespace op {

class StaticShapeNonMaxSuppression : public ngraph::op::v4::NonMaxSuppression {
public:
    static constexpr NodeTypeInfo type_info{"StaticShapeStaticShapeNonMaxSuppression", 0};
    const NodeTypeInfo& get_type_info() const override { return type_info; }
    StaticShapeNonMaxSuppression() = default;

    StaticShapeNonMaxSuppression(const Output<Node>& boxes,
                      const Output<Node>& scores,
                      const Output<Node>& max_output_boxes_per_class,
                      const Output<Node>& iou_threshold,
                      const Output<Node>& score_threshold,
                      const BoxEncodingType box_encoding = BoxEncodingType::CORNER,
                      const bool sort_result_descending = true,
                      const ngraph::element::Type& output_type = ngraph::element::i64);

    StaticShapeNonMaxSuppression(const Output<Node>& boxes,
                      const Output<Node>& scores,
                      const BoxEncodingType box_encoding = BoxEncodingType::CORNER,
                      const bool sort_result_descending = true,
                      const ngraph::element::Type& output_type = ngraph::element::i64);

    void validate_and_infer_types() override;
    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;
};

}  // namespace op
}  // namespace vpu
}  // namespace ngraph
