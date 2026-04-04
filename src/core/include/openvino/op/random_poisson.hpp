// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/op/op.hpp"
#include "openvino/op/util/attr_types.hpp"

namespace ov {
namespace op {
namespace v17 {
/// \brief Tensor RandomPoisson operation.
/// \ingroup ov_ops_cpp_api
class OPENVINO_API RandomPoisson : public Op {
public:
    OPENVINO_OP("RandomPoisson", "opset17");

    RandomPoisson() = default;
    /// \brief Constructs a RandomPoisson operation.
    ///
    /// \param input The input tensor with the rates values.
    /// \param global_seed The global seed value.
    /// \param op_seed The operational seed value.
    /// \param alignment The alignment mode.
    RandomPoisson(const Output<Node>& input,
                  // const Output<Node>& generator, We can ignore the generator input for now
                  uint64_t global_seed = 0,
                  uint64_t op_seed = 0,
                  ov::op::PhiloxAlignment alignment = ov::op::PhiloxAlignment::TENSORFLOW);

    void validate_and_infer_types() override;

    bool visit_attributes(AttributeVisitor& visitor) override;

    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;

    /// \return Turns off constant folding for RandomPoisson operation.
    bool can_constant_fold(const OutputVector& inputs_values) const override;

    /// \return The global seed value.
    uint64_t get_global_seed() const;

    void set_global_seed(uint64_t seed);

    /// \return The operational seed value.
    uint64_t get_op_seed() const;

    void set_op_seed(uint64_t seed2);

    /// \return The state value.
    std::pair<uint64_t, uint64_t> get_state() const;

    /// \brief Set the state value.
    void set_state(std::pair<uint64_t, uint64_t> state) const;

    /// \return The alignment mode.
    ov::op::PhiloxAlignment get_alignment() const;

    void set_alignment(ov::op::PhiloxAlignment alignment);

    bool evaluate(TensorVector& outputs, const TensorVector& inputs) const override;
    bool has_evaluate() const override;

protected:
    uint64_t m_global_seed;
    uint64_t m_op_seed;
    ov::op::PhiloxAlignment m_alignment = ov::op::PhiloxAlignment::TENSORFLOW;

    mutable std::pair<uint64_t, uint64_t> m_state;
    // friend struct random_poisson::Evaluate;
};
}  // namespace v17
}  // namespace op
}  // namespace ov