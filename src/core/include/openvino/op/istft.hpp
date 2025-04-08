// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/op/op.hpp"

namespace ov::op::v16 {
/// \brief An operation ISTFT that computes the Inverse Short Time Fourier Transform.
/// \ingroup ov_ops_cpp_api
class OPENVINO_API ISTFT : public Op {
public:
    OPENVINO_OP("ISTFT", "opset16");
    ISTFT() = default;

    /// \brief Constructs an ISTFT operation with signal length to be inferred
    ///
    /// \param data  Input data
    /// \param window Window values applied in ISTFT
    /// \param frame_size Scalar value representing the size of Fourier Transform
    /// \param frame_step The distance (number of samples) between successive window frames
    /// \param center Flag signaling if the signal input has been padded before STFT
    /// \param normalized Flag signaling if the STFT result has been normalized
    ISTFT(const Output<Node>& data,
          const Output<Node>& window,
          const Output<Node>& frame_size,
          const Output<Node>& frame_step,
          const bool center,
          const bool normalized);

    /// \brief Constructs an ISTFT operation with signal length provided
    ///
    /// \param data  Input data
    /// \param window Window values applied in ISTFT
    /// \param frame_size Scalar value representing the size of Fourier Transform
    /// \param frame_step The distance (number of samples) between successive window frames
    /// \param signal_length The signal length of the original signal
    /// \param center Flag signaling if the signal input has been padded before STFT
    /// \param normalized Flag signaling if the STFT result has been normalized
    ISTFT(const Output<Node>& data,
          const Output<Node>& window,
          const Output<Node>& frame_size,
          const Output<Node>& frame_step,
          const Output<Node>& signal_length,
          const bool center,
          const bool normalized);

    bool visit_attributes(AttributeVisitor& visitor) override;
    void validate_and_infer_types() override;
    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;

    bool get_center() const;
    void set_center(const bool center);

    bool get_normalized() const;

private:
    bool m_center = false;
    bool m_normalized = false;
};
}  // namespace ov::op::v16
