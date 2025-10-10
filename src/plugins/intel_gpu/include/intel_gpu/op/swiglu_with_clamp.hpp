// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/op/op.hpp"
#include "ov_ops/glu.hpp"
#include "openvino/op/util/sub_graph_base.hpp"

namespace ov::intel_gpu::op {

using GluType = ov::op::internal::GLU::GluType;

/// \brief Operator performing Gated Linear Unit Activation
/// This operation performs gated linear unit activation that combines swish or gelu activation function
class SwiGluWithClamp : public ov::op::Op {
public:
    OPENVINO_OP("SwiGluWithClamp", "ie_internal_opset");

    SwiGluWithClamp() = default;
    /// \brief Constructs an SwiGluWithClamp operation.
    ///
    /// \param data Input tensor with data
    /// \param axis The index of an axis in "data" along which to perform the split
    /// \param split_lenghts A list containing the sizes of each output tensor along the split "axis"
    /// \param split_to_glu_idx Output index of variadic split, which is connected to GLU
    /// \param clamp_min Clamp output more than min value
    /// \parma clamp_max Clamp output less than max min
    /// \param output_type Output element type
    SwiGluWithClamp(const ov::Output<Node>& data,
        int64_t axis,
        int64_t split_lengths,
        const GluType glu_type,
        const size_t split_to_glu_idx,
        const double clamp_min,
        const double clamp_max,
        const ov::element::Type output_type = ov::element::dynamic);

    void validate_and_infer_types() override;

    std::shared_ptr<Node> clone_with_new_inputs(const ov::OutputVector& new_args) const override;

    int64_t get_axis() const {
        return m_axis;
    }
    int64_t get_split_lengths() const {
        return m_split_lengths;
    }
    GluType get_glu_type() const {
        return m_glu_type;
    }
    size_t get_split_to_glu_idx() const {
        return m_split_to_glu_idx;
    }
    int64_t get_clamp_min() const {
        return m_clamp_min;
    }
    int64_t get_clamp_max() const {
        return m_clamp_max;
    }

    void set_axis(int64_t axis) {
        m_axis = axis;
    }
    void set_split_lengths(int64_t split_lengths) {
        m_split_lengths = split_lengths;
    }
    void set_glu_type(GluType glu_type) {
        m_glu_type = glu_type;
    }
    void set_split_to_glu_idx(size_t split_to_glu_idx) {
        m_split_to_glu_idx = split_to_glu_idx;
    }
    void set_clamp_min(int64_t min) {
        m_clamp_min = min;
    }
    void set_clamp_max(int64_t max) {
        m_clamp_max = max;
    }

private:
    int64_t m_axis = 0;
    int64_t m_split_lengths = 0;
    GluType m_glu_type = GluType::Swish;
    size_t m_split_to_glu_idx = 0;
    double m_clamp_min = 0;
    double m_clamp_max = 0;
    ov::element::Type m_output_type{};
};

}  // namespace ov::intel_gpu::op
