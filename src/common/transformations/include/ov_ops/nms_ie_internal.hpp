// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <string>

#include "openvino/core/coordinate_diff.hpp"
#include "openvino/op/op.hpp"
#include "transformations_visibility.hpp"

namespace ov {
namespace op {
namespace internal {

class TRANSFORMATIONS_API NonMaxSuppressionIEInternal : public Op {
public:
    OPENVINO_OP("NonMaxSuppressionIEInternal", "ie_internal_opset");

    NonMaxSuppressionIEInternal() = default;

    static constexpr int Rotation_None = 0;
    static constexpr int Rotation_Clockwise = 1;
    static constexpr int Rotation_Counterclockwise = 2;

    NonMaxSuppressionIEInternal(const Output<Node>& boxes,
                                const Output<Node>& scores,
                                const Output<Node>& max_output_boxes_per_class,
                                const Output<Node>& iou_threshold,
                                const Output<Node>& score_threshold,
                                int center_point_box,
                                bool sort_result_descending,
                                const element::Type& output_type = element::i64,
                                const element::Type& score_output_type = element::f32,
                                const int rotation = Rotation_None);

    NonMaxSuppressionIEInternal(const Output<Node>& boxes,
                                const Output<Node>& scores,
                                const Output<Node>& max_output_boxes_per_class,
                                const Output<Node>& iou_threshold,
                                const Output<Node>& score_threshold,
                                const Output<Node>& soft_nms_sigma,
                                int center_point_box,
                                bool sort_result_descending,
                                const element::Type& output_type = element::i64,
                                const element::Type& score_output_type = element::f32,
                                const int rotation = Rotation_None);

    void validate_and_infer_types() override;

    bool visit_attributes(AttributeVisitor& visitor) override;

    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;

    int m_center_point_box;
    bool m_sort_result_descending = true;
    element::Type m_output_type;
    element::Type m_scores_output_type;
    int m_rotation{Rotation_None};

private:
    int64_t max_boxes_output_from_input() const;
};

}  // namespace internal
}  // namespace op
}  // namespace ov
