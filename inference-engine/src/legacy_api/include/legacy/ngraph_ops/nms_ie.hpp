// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <string>

#include <ie_api.h>

#include "ngraph/op/op.hpp"

namespace ngraph {
namespace op {

class INFERENCE_ENGINE_API_CLASS(NonMaxSuppressionIE) : public Op {
public:
    static constexpr NodeTypeInfo type_info{"NonMaxSuppressionIE", 1};
    const NodeTypeInfo& get_type_info() const override { return type_info; }

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

class INFERENCE_ENGINE_API_CLASS(NonMaxSuppressionIE2) : public NonMaxSuppressionIE {
public:
    static constexpr NodeTypeInfo type_info{"NonMaxSuppressionIE", 2};
    const NodeTypeInfo& get_type_info() const override { return type_info; }

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

class INFERENCE_ENGINE_API_CLASS(NonMaxSuppressionIE3) : public Op {
public:
    NGRAPH_RTTI_DECLARATION;

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

}  // namespace op
}  // namespace ngraph
