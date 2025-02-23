// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/op/op.hpp"

namespace ov::op::v16 {
/// \brief An operation which computes the maximum values along segments of a tensor.
/// \ingroup ov_ops_cpp_api
class OPENVINO_API SegmentMax : public ov::op::Op {
public:
    OPENVINO_OP("SegmentMax", "opset16", ov::op::Op);

    SegmentMax() = default;

    /// \brief Constructs a SegmentMax operation.
    ///
    /// \param data Input tensor with data
    /// \param segment_ids Indices of segments in the data input tensor
    /// \param fill_mode The value assigned to segments which are empty
    SegmentMax(const Output<Node>& data, const Output<Node>& segment_ids, const op::FillMode fill_mode);

    /// \brief Constructs a SegmentMax operation.
    ///
    /// \param data Input tensor with data
    /// \param segment_ids Indices of segments in the data input tensor
    /// \param num_segments The segments count
    /// \param fill_mode The value assigned to segments which are empty
    SegmentMax(const Output<Node>& data,
               const Output<Node>& segment_ids,
               const Output<Node>& num_segments,
               const op::FillMode fill_mode);

    bool visit_attributes(AttributeVisitor& visitor) override;
    void validate_and_infer_types() override;
    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;

    const op::FillMode get_fill_mode() const;

private:
    op::FillMode m_fill_mode{};
};

}  // namespace ov::op::v16
