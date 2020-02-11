// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>

#include "ngraph/node.hpp"
#include "ngraph/op/op.hpp"
#include "ngraph/op/util/fused_op.hpp"

namespace ngraph {
namespace op {

/// \brief Operator performing Matrix Multiplication.
class GemmIE : public Op {
public:
    static constexpr NodeTypeInfo type_info{"GemmIE", 0};
    const NodeTypeInfo& get_type_info() const override { return type_info; }
    GemmIE() = default;
    /// \brief Constructs an ScaleShift operation.
    ///
    /// \param A Matrix A
    /// \param B Matrix B
    /// \param C Matrix C
    GemmIE(const Output<Node> & A,
             const Output<Node> & B,
             const bool & transpose_a,
             const bool & transpose_b,
             const Shape & output_shape);

    void validate_and_infer_types() override;

    std::shared_ptr<Node> copy_with_new_args(const NodeVector& new_args) const override;

    bool get_transpose_a() const { return m_transpose_a; }
    bool get_transpose_b() const { return m_transpose_b; }

private:
    Shape m_output_shape;
    bool m_transpose_a = false;
    bool m_transpose_b = false;
};

}  // namespace op
}  // namespace ngraph
