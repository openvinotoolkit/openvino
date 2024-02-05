// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/op/op.hpp"

namespace ov {
namespace op {
namespace v14 {
/// \brief Inverse operation computes the inverse of the input tensor.
///
/// \ingroup ov_ops_cpp_api
class OPENVINO_API Inverse : public Op {
public:
    OPENVINO_OP("Inverse", "opset14");
    Inverse() = default;
    /**
     * @brief Inverse operation computes the inverse of the input tensor.
     *
     * @param input Input matrix to compute the inverse for.
     * @param adjoint Boolean that determines whether to return a normal inverse or adjoint (conjugate transpose) of the
     * input matrix.
     */
    Inverse(const Output<Node>& input, const bool adjoint = false);

    bool visit_attributes(AttributeVisitor& visitor) override;
    void validate_and_infer_types() override;
    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;

    bool get_adjoint() const;
    void set_adjoint(const bool adjoint);

private:
    bool m_adjoint;
};
}  // namespace v14
}  // namespace op
}  // namespace ov
