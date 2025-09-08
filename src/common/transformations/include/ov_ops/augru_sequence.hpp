// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <string>
#include <vector>

#include "openvino/op/util/rnn_cell_base.hpp"
#include "transformations_visibility.hpp"

namespace ov {
namespace op {
namespace internal {
///
/// \brief AUGRUSequence operation.
///
/// \ingroup ov_ops_cpp_api
class TRANSFORMATIONS_API AUGRUSequence : public ov::op::util::RNNCellBase {
public:
    OPENVINO_OP("AUGRUSequence", "ie_internal_opset", ov::op::util::RNNCellBase);

    AUGRUSequence();
    AUGRUSequence(const Output<Node>& X,
                  const Output<Node>& H_t,
                  const Output<Node>& sequence_lengths,
                  const Output<Node>& W,
                  const Output<Node>& R,
                  const Output<Node>& B,
                  const Output<Node>& A,
                  size_t hidden_size);

    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;

    void validate_and_infer_types() override;

    bool visit_attributes(AttributeVisitor& visitor) override;
    bool get_linear_before_reset() const {
        return m_linear_before_reset;
    }
    op::RecurrentSequenceDirection get_direction() const {
        return m_direction;
    }
    void set_direction(const RecurrentSequenceDirection& direction) {
        m_direction = direction;
    }

protected:
    op::RecurrentSequenceDirection m_direction;
    bool m_linear_before_reset;
};
}  // namespace internal
}  // namespace op
}  // namespace ov
