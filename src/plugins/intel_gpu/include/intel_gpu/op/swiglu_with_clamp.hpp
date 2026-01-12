// Copyright (C) 2018-2026 Intel Corporation
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
    /// \param glu_stride Size of stride for gate index
    /// \param glu_type Either of Swish / Gelu / Gelu_Tanh
    /// \param gate_idx Output index to apply gating activation
    /// \param clamp_min Clamp output more than min value
    /// \param clamp_max Clamp output less than max min
    /// \param swiglu_beta Swiglu beta (default : 1.0f)
    /// \param up_add_val Value added to up (default : 0.0f)
    /// \param output_type Output element type
    SwiGluWithClamp(const ov::Output<Node>& data,
        int64_t axis,
        int64_t glu_stride,
        const GluType glu_type,
        const size_t gate_idx,
        const float clamp_min,
        const float clamp_max,
        const float swiglu_beta = 1.0f,
        const float up_add_val = 0.0f,
        const ov::element::Type output_type = ov::element::dynamic);

    void validate_and_infer_types() override;

    std::shared_ptr<Node> clone_with_new_inputs(const ov::OutputVector& new_args) const override;

    int64_t get_axis() const {
        return m_axis;
    }
    int64_t get_glu_stride() const {
        return m_glu_stride;
    }
    GluType get_glu_type() const {
        return m_glu_type;
    }
    size_t get_gate_idx() const {
        return m_gate_idx;
    }
    float get_clamp_min() const {
        return m_clamp_min;
    }
    float get_clamp_max() const {
        return m_clamp_max;
    }
    float get_swiglu_beta() const {
        return m_swiglu_beta;
    }
    float get_up_add_val() const {
        return m_up_add_val;
    }
    void set_axis(int64_t axis) {
        m_axis = axis;
    }
    void set_glu_stride(int64_t stride) {
        m_glu_stride = stride;
    }
    void set_glu_type(GluType glu_type) {
        m_glu_type = glu_type;
    }
    void set_gate_idx(size_t gate_idx) {
        m_gate_idx = gate_idx;
    }
    void set_clamp_min(int64_t min) {
        m_clamp_min = min;
    }
    void set_clamp_max(int64_t max) {
        m_clamp_max = max;
    }
    void set_swiglu_beta(float beta) {
        m_swiglu_beta = beta;
    }
    void set_up_add_val(float val) {
        m_up_add_val = val;
    }

private:
    int64_t m_axis = 0;
    int64_t m_glu_stride = 0;
    GluType m_glu_type = GluType::Swish;
    size_t m_gate_idx = 0;
    float m_clamp_min = std::numeric_limits<float>::lowest();
    float m_clamp_max = std::numeric_limits<float>::max();
    float m_swiglu_beta = 1.0f;
    float m_up_add_val = 0.0f;
    ov::element::Type m_output_type{};
};

std::vector<ov::PartialShape> shape_infer(const ov::intel_gpu::op::SwiGluWithClamp* op,
                                          const std::vector<ov::PartialShape>& input_shapes);

}  // namespace ov::intel_gpu::op
