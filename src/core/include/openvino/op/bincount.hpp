// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstdint>
#include <memory>

#include "openvino/op/op.hpp"

namespace ov {
namespace op {
namespace v17 {
/// \brief Counts the number of occurrences of each value in a 1-D integer tensor.
///
/// \ingroup ov_ops_cpp_api
class OPENVINO_API Bincount : public Op {
public:
    OPENVINO_OP("Bincount", "opset17", op::Op);

    Bincount() = default;

    /// \brief Constructs a Bincount operation without weights (result type is i64).
    ///
    /// \param data       1-D non-negative integer tensor
    /// \param minlength  Minimum length of output; defaults to 0
    Bincount(const Output<Node>& data, int64_t minlength = 0);

    /// \brief Constructs a Bincount operation with weights.
    ///
    /// \param data       1-D non-negative integer tensor
    /// \param weights    1-D float/integer tensor (same length as data)
    /// \param minlength  Minimum length of output; defaults to 0
    Bincount(const Output<Node>& data, const Output<Node>& weights, int64_t minlength = 0);

    bool visit_attributes(AttributeVisitor& visitor) override;
    void validate_and_infer_types() override;
    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;
    bool evaluate(TensorVector& outputs, const TensorVector& inputs) const override;
    bool has_evaluate() const override;

    int64_t get_minlength() const {
        return m_minlength;
    }
    void set_minlength(int64_t minlength) {
        m_minlength = minlength;
    }

private:
    int64_t m_minlength = 0;
};
}  // namespace v17
}  // namespace op
}  // namespace ov
