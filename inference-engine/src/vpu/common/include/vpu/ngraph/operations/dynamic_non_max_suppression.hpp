// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ngraph/op/non_max_suppression.hpp"

namespace ngraph { namespace vpu { namespace op {

class DynamicNonMaxSuppression : public ngraph::op::v4::NonMaxSuppression {
public:
    static constexpr NodeTypeInfo type_info{"DynamicNonMaxSuppression", 0};
    const NodeTypeInfo& get_type_info() const override { return type_info; }
    DynamicNonMaxSuppression() = default;

    /// \brief Constructs a DynamicNonMaxSuppression operation.
    ///
    /// \param boxes Node producing the box coordinates
    /// \param scores Node producing the box scores
    /// \param max_output_boxes_per_class Node producing maximum number of boxes to be
    /// selected per class
    /// \param iou_threshold Node producing intersection over union threshold
    /// \param score_threshold Node producing minimum score threshold
    /// \param box_encoding Specifies the format of boxes data encoding
    /// \param sort_result_descending Specifies whether it is necessary to sort selected
    /// boxes across batches
    /// \param output_type Specifies the output tensor type
    DynamicNonMaxSuppression(const Output<Node>& boxes,
                             const Output<Node>& scores,
                             const Output<Node>& max_output_boxes_per_class,
                             const Output<Node>& iou_threshold,
                             const Output<Node>& score_threshold,
                             const BoxEncodingType box_encoding = BoxEncodingType::CORNER,
                             const bool sort_result_descending = true,
                             const ngraph::element::Type& output_type = ngraph::element::i64);

    /// \brief Constructs a DynamicNonMaxSuppression operation with default values for the last
    ///        3 inputs
    ///
    /// \param boxes Node producing the box coordinates
    /// \param scores Node producing the box coordinates
    /// \param box_encoding Specifies the format of boxes data encoding
    /// \param sort_result_descending Specifies whether it is necessary to sort selected
    /// boxes across batches
    /// \param output_type Specifies the output tensor type
    DynamicNonMaxSuppression(const Output<Node>& boxes,
                             const Output<Node>& scores,
                             const BoxEncodingType box_encoding = BoxEncodingType::CORNER,
                             const bool sort_result_descending = true,
                             const ngraph::element::Type& output_type = ngraph::element::i64);

    void validate_and_infer_types() override;

    std::shared_ptr<Node>
    clone_with_new_inputs(const OutputVector& new_args) const override;
};

}  // namespace op
}  // namespace vpu
}  // namespace ngraph
