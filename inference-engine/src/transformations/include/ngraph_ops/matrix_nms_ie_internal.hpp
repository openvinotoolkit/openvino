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

class TRANSFORMATIONS_API MatrixNmsIEInternal : public Op {
public:
    static constexpr NodeTypeInfo type_info{"MatrixNmsIEInternal", 0};
    const NodeTypeInfo& get_type_info() const override { return type_info; }

    MatrixNmsIEInternal(const Output<Node>& boxes,
                        const Output<Node>& scores,
                        const int32_t sort_result_type = 2,
                        const bool sort_result_across_batch = true,
                        const ngraph::element::Type& output_type = ngraph::element::i64,
                        const float score_threshold = 0.0f,
                        const int nms_top_k = -1,
                        const int keep_top_k = -1,
                        const int background_class = -1,
                        const int32_t decay_function = 1,
                        const float gaussian_sigma = 2.0f,
                        const float post_threshold = 0.0f,
                        const bool normalized = true);

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
    float m_score_threshold;
    int m_nms_top_k;
    int m_keep_top_k;
    int m_background_class;
    int32_t m_decay_function;
    float m_gaussian_sigma;
    float m_post_threshold;
    bool m_normalized;
};

}  // namespace internal
}  // namespace op
}  // namespace ngraph
