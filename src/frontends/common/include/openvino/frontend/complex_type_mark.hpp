// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/core/type/element_type.hpp"
#include "openvino/frontend/exception.hpp"
#include "openvino/frontend/node_context.hpp"
#include "openvino/frontend/visibility.hpp"
#include "openvino/op/util/framework_node.hpp"

namespace ov {
namespace frontend {

// ComplexTypeMark serves to mark places that require complex type propagation
// that means to represent native complex type with simulating floating-point tensor
// that has one extra dimension to concatenate real and imaginary parts of complex tensor.
// For example, a tensor of complex type with shape [N1, N2, ..., Nk] will be transformed
// into a floating-point tensor [N1, N2, ..., Nk, 2]
// where a slice with index [..., 0] represents a real part and
// a slice with index [..., 1] represents a imaginary part.
class FRONTEND_API ComplexTypeMark : public ov::op::util::FrameworkNode {
public:
    OPENVINO_OP("ComplexTypeMark", "util", ov::op::util::FrameworkNode);

    ComplexTypeMark(const ov::Output<ov::Node>& input,
                    const ov::element::Type& complex_part_type = ov::element::dynamic);

    ComplexTypeMark(const ov::Output<ov::Node>& real,
                    const ov::Output<ov::Node>& imag,
                    const ov::element::Type& complex_part_type = ov::element::dynamic);

    ~ComplexTypeMark() override;

    void validate_and_infer_types() override {
        set_output_type(0, ov::element::dynamic, PartialShape::dynamic());
    }

    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& inputs) const override {
        auto complex_type_mark = std::make_shared<ComplexTypeMark>(inputs[0], m_complex_part_type);
        complex_type_mark->set_attrs(get_attrs());
        return complex_type_mark;
    }

    ov::element::Type get_complex_part_type() const {
        return m_complex_part_type;
    }

    // Get a real part of the complex tensor
    ov::Output<ov::Node> get_real(bool squeezed = true);

    // Get an imaginary part of the complex tensor
    ov::Output<ov::Node> get_imag(bool squeezed = true);

    // Get floating-point representation of the complex tensor
    ov::Output<ov::Node> get_data();

    // Compute summation of two operands that can be of complex type
    // if operand is of complex type, complex type will be indicated by bool flag
    // complex tensor is represented as a real tensor with auxiliary dimension 2 in the tail
    // types of both operands must be aligned prior to the call
    static ov::Output<ov::Node> add(const NodeContext& context,
                                    const ov::Output<ov::Node>& lhs,
                                    const ov::Output<ov::Node>& rhs);

    // Compute subtraction of two operands that can be of complex type
    // if operand is of complex type, complex type will be indicated by bool flag
    // complex tensor is represented as a real tensor with auxiliary dimension 2 in the tail
    // types of both operands must be aligned prior to the call
    static ov::Output<ov::Node> sub(const NodeContext& context,
                                    const ov::Output<ov::Node>& lhs,
                                    const ov::Output<ov::Node>& rhs);

    // Compute multiplication of two operands that can be of complex type
    // if operand is of complex type, complex type will be indicated by bool flag
    // complex tensor is represented as a real tensor with auxiliary dimension 2 in the tail
    // types of both operands must be aligned prior to the call
    static ov::Output<ov::Node> mul(const NodeContext& context,
                                    const ov::Output<ov::Node>& lhs,
                                    const ov::Output<ov::Node>& rhs);

    // Compute inverse of operand that can be of complex type
    // if operand is of complex type, complex type will be indicated by bool flag
    // complex tensor is represented as a real tensor with auxiliary dimension 2 in the tail
    static ov::Output<ov::Node> inv(const NodeContext& context, const ov::Output<ov::Node>& data);

    // Compute division of two operands that can be of complex type
    // if operand is of complex type, complex type will be indicated by bool flag
    // complex tensor is represented as a real tensor with auxiliary dimension 2 in the tail
    // types of both operands must be aligned prior to the call
    static ov::Output<ov::Node> div(const NodeContext& context,
                                    const ov::Output<ov::Node>& lhs,
                                    const ov::Output<ov::Node>& rhs);

    // Convert type of real and imaginary parts of input to like type
    static ov::Output<ov::Node> convert_like(const NodeContext& context,
                                             const ov::Output<ov::Node>& input,
                                             const ov::Output<ov::Node>& like);

    // Compute abs of operand that can be of complex type
    static ov::Output<ov::Node> abs(const NodeContext& context, const ov::Output<ov::Node>& data);

    // Compute exp of operand that can be of complex type
    static ov::Output<ov::Node> exp(const NodeContext& context, const ov::Output<ov::Node>& data);

private:
    ov::element::Type m_complex_part_type;

    // floating-point tensor that represents complex tensor
    ov::Output<ov::Node> m_data;

    // real part of the complex tensor in squeezed form (no auxiliary dimension)
    ov::Output<ov::Node> m_real;

    // imaginary part of the complex tensor in squeezed form (no auxiliary dimension)
    ov::Output<ov::Node> m_imag;
};

}  // namespace frontend
}  // namespace ov
