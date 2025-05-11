// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <string>
#include <vector>

#include "openvino/op/util/activation_functions.hpp"
#include "openvino/op/util/rnn_cell_base.hpp"
#include "transformations_visibility.hpp"

namespace ov {
namespace op {
namespace internal {
///
/// \brief AUGRUCell operation.
///
/// \ingroup ov_ops_cpp_api
class TRANSFORMATIONS_API AUGRUCell : public ov::op::util::RNNCellBase {
public:
    OPENVINO_OP("AUGRUCell", "ie_internal_opset", ov::op::util::RNNCellBase);

    AUGRUCell();
    AUGRUCell(const Output<Node>& X,
              const Output<Node>& initial_hidden_state,
              const Output<Node>& W,
              const Output<Node>& R,
              const Output<Node>& B,
              const Output<Node>& A,
              size_t hidden_size);

    void validate_and_infer_types() override;
    bool visit_attributes(AttributeVisitor& visitor) override;
    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;

    bool get_linear_before_reset() const {
        return m_linear_before_reset;
    }

private:
    ///
    /// \brief The Activation function f.
    ///
    util::ActivationFunction m_activation_f;
    ///
    /// \brief The Activation function g.
    ///
    util::ActivationFunction m_activation_g;

    static constexpr std::size_t s_gates_count{3};
    ///
    /// \brief Control whether or not apply the linear transformation.
    ///
    /// \note The linear transformation may be applied when computing the output of
    ///       hidden gate. It's done before multiplying by the output of the reset gate.
    ///
    bool m_linear_before_reset = false;
};
}  // namespace internal
}  // namespace op
}  // namespace ov
