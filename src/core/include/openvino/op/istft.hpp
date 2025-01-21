// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/op/op.hpp"

namespace ov {
namespace op {
namespace v16 {
/// \brief An operation ISTFT that computes the Inverse Short Time Fourier Transform.
/// \ingroup ov_ops_cpp_api
class OPENVINO_API ISTFT : public Op {
public:
    OPENVINO_OP("ISTFT", "opset16");
    ISTFT() = default;

    /// \brief Constructs an ISTFT operation.
    ///
    /// \param data  Input data
    /// \param window Window values applied in STFT
    /// \param frame_size Scalar value representing the size of Fourier Transform
    /// \param frame_step The distance (number of samples) between successive window frames
    /// \param length The length of the original signal
    /// \param center Flag signaling if the signal input has been padded before STFT
    /// \param normalized Flag signaling if the STFT result has been normalized.
    ISTFT(const Output<Node>& data,
          const Output<Node>& window,
          const Output<Node>& frame_size,
          const Output<Node>& frame_step,
          const Output<Node>& length,
          const bool center,
          const bool normalized);

    bool visit_attributes(AttributeVisitor& visitor) override;
    void validate_and_infer_types() override;
    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;

    bool get_center() const;

    bool evaluate(TensorVector& outputs, const TensorVector& inputs) const override;
    bool has_evaluate() const override;

private:
    bool m_center = false;
    bool m_normalized = false;
};
}  // namespace v16
}  // namespace op
}  // namespace ov
