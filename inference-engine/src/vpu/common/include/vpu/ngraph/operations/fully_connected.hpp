// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ie_api.h>

#include "ngraph/node.hpp"
#include "ngraph/op/op.hpp"

namespace ngraph {
namespace vpu {
namespace op {

/// \brief Operator performing Matrix Multiplication.
class FullyConnected : public ngraph::op::Op {
public:
    static constexpr NodeTypeInfo type_info{"FullyConnected", 0};
    const NodeTypeInfo& get_type_info() const override { return type_info; }
    FullyConnected() = default;
    /// \brief Constructs an FullyConnected operation.
    ///
    /// \param A Matrix A
    /// \param B Matrix B
    /// \param C Matrix C
    FullyConnected(const Output<Node> & A,
                   const Output<Node> & B,
                   const Output<Node> & C,
                   const Shape & output_shape,
                   const element::Type output_type = element::undefined);

    bool visit_attributes(AttributeVisitor &visitor) override;

    void validate_and_infer_types() override;

    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;

    size_t get_out_size() const { return m_output_size; }

    element::Type get_output_type() const { return m_output_type; }

private:
    size_t m_output_size = 0;
    Shape m_output_shape = {};
    element::Type m_output_type;
};

}  // namespace op
}  // namespace vpu
}  // namespace ngraph
