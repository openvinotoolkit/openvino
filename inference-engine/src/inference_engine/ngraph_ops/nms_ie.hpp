// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <string>

#include "ngraph/op/op.hpp"

namespace ngraph {
namespace op {

class NonMaxSuppressionIE : public Op {
public:
    static constexpr NodeTypeInfo type_info{"NonMaxSuppressionIE", 1};
    const NodeTypeInfo& get_type_info() const override { return type_info; }

    NonMaxSuppressionIE(const Output<Node>& boxes,
                        const Output<Node>& scores,
                        const Output<Node>& max_output_boxes_per_class,
                        const Output<Node>& iou_threshold,
                        const Output<Node>& score_threshold,
                        const Shape& output_shape,
                        int center_point_box,
                        bool sort_result_descending);

    void validate_and_infer_types() override;

    std::shared_ptr<Node> copy_with_new_args(const NodeVector& new_args) const override;

    int m_center_point_box;
    bool m_sort_result_descending = true;
    Shape m_output_shape;
};

}  // namespace op
}  // namespace ngraph
