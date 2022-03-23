// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/op/op.hpp"

namespace ov {
namespace op {
namespace v8 {
/// \brief Tensor RandomUniform operation.
/// \ingroup ov_ops_cpp_api
class OPENVINO_API RandomUniform : public Op {
public:
    OPENVINO_OP("RandomUniform", "opset8");
    BWDCMP_RTTI_DECLARATION;

    RandomUniform() = default;

    ///
    /// \brief      Constructs a RandomUniform operation.
    ///
    /// \param      out_shape         Node producing the tensor with output shape.
    /// \param      min_val           Node producing the tensor with minimum value.
    /// \param      max_val           Node producing the tensor with maximum value.
    /// \param      out_type          Output type of the tensor.
    /// \param      global_seed       Global seed value.
    /// \param      op_seed           Operational seed value.
    RandomUniform(const Output<Node>& out_shape,
                  const Output<Node>& min_val,
                  const Output<Node>& max_val,
                  const ov::element::Type& out_type,
                  uint64_t global_seed = 0,
                  uint64_t op_seed = 0);

    void validate_and_infer_types() override;

    bool visit_attributes(AttributeVisitor& visitor) override;

    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;

    /// \return Turns off constant folding for RandomUniform operation.
    bool constant_fold(OutputVector& output_values, const OutputVector& inputs_values) override {
        return false;
    }

    /// \return The output tensor type.
    const ov::element::Type& get_out_type() const {
        return m_output_type;
    }
    void set_out_type(const ov::element::Type& output_type) {
        m_output_type = output_type;
    }

    /// \return The global seed value.
    uint64_t get_global_seed() const {
        return m_global_seed;
    }
    void set_global_seed(uint64_t seed) {
        m_global_seed = seed;
    }

    /// \return The operational seed value.
    uint64_t get_op_seed() const {
        return m_op_seed;
    }
    void set_op_seed(uint64_t seed2) {
        m_op_seed = seed2;
    }

    /// \return The state value.
    std::pair<uint64_t, uint64_t> get_state() const {
        return m_state;
    }

    OPENVINO_SUPPRESS_DEPRECATED_START
    bool evaluate(const HostTensorVector& outputs, const HostTensorVector& inputs) const override;
    OPENVINO_SUPPRESS_DEPRECATED_END

    bool has_evaluate() const override;

protected:
    ov::element::Type m_output_type;
    uint64_t m_global_seed;
    uint64_t m_op_seed;

    mutable std::mutex m_state_mutex;
    mutable std::pair<uint64_t, uint64_t> m_state;
};
}  // namespace v8
}  // namespace op
}  // namespace ov
