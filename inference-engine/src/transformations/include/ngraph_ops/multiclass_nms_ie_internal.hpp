// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <string>

#include <transformations_visibility.hpp>

#include "ngraph/coordinate_diff.hpp"
#include "ngraph/op/op.hpp"

namespace ngraph {
namespace op {
namespace internal {

class TRANSFORMATIONS_API MulticlassNmsIEInternal : public Op {
public:
    static constexpr NodeTypeInfo type_info{"MulticlassNmsIEInternal", 0};
    const NodeTypeInfo& get_type_info() const override { return type_info; }

    MulticlassNmsIEInternal(const Output<Node>& boxes,
                            const Output<Node>& scores,
                            const int32_t sort_result_type = 2,
                            const bool sort_result_across_batch = true,
                            const ngraph::element::Type& output_type = ngraph::element::i32,
                            const float iou_threshold = 0.0f,
                            const float score_threshold = 0.0f,
                            const int nms_top_k = -1,
                            const int keep_top_k = -1,
                            const int background_class = -1,
                            const float nms_eta = 1.0f);

    void validate_and_infer_types() override;

    bool visit_attributes(AttributeVisitor& visitor) override;

    // TODO: test usage only
    bool evaluate(const HostTensorVector& outputs,
                                    const HostTensorVector& inputs) const override;
    bool has_evaluate() const override { return true; }

    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector & new_args) const override;

    int32_t m_sort_result_type;
    bool m_sort_result_across_batch;
    ngraph::element::Type m_output_type;
    float m_iou_threshold;
    float m_score_threshold;
    int m_nms_top_k;
    int m_keep_top_k;
    int m_background_class;
    float m_nms_eta;
};

}  // namespace internal
}  // namespace op
}  // namespace ngraph
