// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <string>

#include <ie_api.h>

#include "ngraph/op/op.hpp"

namespace ngraph {
namespace op {

class NonMaxSuppressionIE;
class NonMaxSuppressionIE2;
class NonMaxSuppressionIE3;

}  // namespace op
}  // namespace ngraph

class ngraph::op::NonMaxSuppressionIE : public Op {
public:
    OPENVINO_OP("NonMaxSuppressionIE", "legacy");
    BWDCMP_RTTI_DECLARATION;

    NonMaxSuppressionIE(const Output<Node>& boxes,
                        const Output<Node>& scores,
                        const Output<Node>& max_output_boxes_per_class,
                        const Output<Node>& iou_threshold,
                        const Output<Node>& score_threshold,
                        int center_point_box,
                        bool sort_result_descending,
                        const ngraph::element::Type& output_type = ngraph::element::i64);

    void validate_and_infer_types() override;

    bool visit_attributes(AttributeVisitor& visitor) override;

    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector & new_args) const override;

    int m_center_point_box;
    bool m_sort_result_descending = true;
    element::Type m_output_type;
};

class ngraph::op::NonMaxSuppressionIE2 : public NonMaxSuppressionIE {
public:
    OPENVINO_OP("NonMaxSuppressionIE2", "legacy");
    BWDCMP_RTTI_DECLARATION;

    NonMaxSuppressionIE2(const Output<Node>& boxes,
                        const Output<Node>& scores,
                        const Output<Node>& max_output_boxes_per_class,
                        const Output<Node>& iou_threshold,
                        const Output<Node>& score_threshold,
                        int center_point_box,
                        bool sort_result_descending,
                        const ngraph::element::Type& output_type = ngraph::element::i64);

    void validate_and_infer_types() override;

    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector & new_args) const override;
};

class ngraph::op::NonMaxSuppressionIE3 : public Op {
public:
    OPENVINO_OP("NonMaxSuppressionIE3", "legacy");
    BWDCMP_RTTI_DECLARATION;

    NonMaxSuppressionIE3(const Output<Node>& boxes,
                         const Output<Node>& scores,
                         const Output<Node>& max_output_boxes_per_class,
                         const Output<Node>& iou_threshold,
                         const Output<Node>& score_threshold,
                         int center_point_box,
                         bool sort_result_descending,
                         const ngraph::element::Type& output_type = ngraph::element::i64);

    NonMaxSuppressionIE3(const Output<Node>& boxes,
                         const Output<Node>& scores,
                         const Output<Node>& max_output_boxes_per_class,
                         const Output<Node>& iou_threshold,
                         const Output<Node>& score_threshold,
                         const Output<Node>& soft_nms_sigma,
                         int center_point_box,
                         bool sort_result_descending,
                         const ngraph::element::Type& output_type = ngraph::element::i64);

    void validate_and_infer_types() override;

    bool visit_attributes(AttributeVisitor& visitor) override;

    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector & new_args) const override;

    int m_center_point_box;
    bool m_sort_result_descending = true;
    element::Type m_output_type;

private:
    int64_t max_boxes_output_from_input() const;
};
