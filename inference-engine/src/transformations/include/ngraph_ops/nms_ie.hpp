// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <string>

#include <transformations_visibility.hpp>

#include "ngraph/op/op.hpp"

namespace ngraph {
namespace op {

class TRANSFORMATIONS_API NonMaxSuppressionIE : public Op {
public:
    static constexpr NodeTypeInfo type_info{"NonMaxSuppressionIE", 1};
    const NodeTypeInfo& get_type_info() const override { return type_info; }

    NonMaxSuppressionIE(const Output<Node>& boxes,
                        const Output<Node>& scores,
                        const Output<Node>& max_output_boxes_per_class,
                        const Output<Node>& iou_threshold,
                        const Output<Node>& score_threshold,
                        int center_point_box,
                        bool sort_result_descending);

    void validate_and_infer_types() override;

    bool visit_attributes(AttributeVisitor& visitor) override;

    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector & new_args) const override;

    int m_center_point_box;
    bool m_sort_result_descending = true;
};

class TRANSFORMATIONS_API NonMaxSuppressionIE2 : public NonMaxSuppressionIE {
public:
    static constexpr NodeTypeInfo type_info{"NonMaxSuppressionIE", 2};
    const NodeTypeInfo& get_type_info() const override { return type_info; }

    NonMaxSuppressionIE2(const Output<Node>& boxes,
                        const Output<Node>& scores,
                        const Output<Node>& max_output_boxes_per_class,
                        const Output<Node>& iou_threshold,
                        const Output<Node>& score_threshold,
                        int center_point_box,
                        bool sort_result_descending);

    void validate_and_infer_types() override;

    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector & new_args) const override;

    int m_center_point_box;
    bool m_sort_result_descending = true;
};

}  // namespace op
}  // namespace ngraph
