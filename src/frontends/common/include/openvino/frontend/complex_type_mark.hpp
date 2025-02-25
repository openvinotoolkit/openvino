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

    ComplexTypeMark(const ov::Output<ov::Node>& input, const ov::element::Type& complex_part_type)
        : ov::op::util::FrameworkNode(ov::OutputVector{input}, 1),
          m_complex_part_type(complex_part_type) {
        validate_and_infer_types();
    }

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

    // Complex data is represented as a floating-point tensor of shape [N1, N2, ..., Nk, 2]
    // where real part is placed by index [..., 0] and
    // `squeeze` flag indicated if squeezing the last dimension is needed
    static ov::Output<ov::Node> get_real_part(const NodeContext& context,
                                              const ov::Output<ov::Node>& complex_data,
                                              bool squeezed = true) {
        return get_complex_part_by_index(context, complex_data, 0, squeezed);
    }

    // Complex tensor is represented as a floating-point tensor of shape [N1, N2, ..., Nk, 2]
    // where imaginary part is placed by index [..., 1] and
    // `squeeze` flag indicated if squeezing the last dimension is needed
    static ov::Output<ov::Node> get_imag_part(const NodeContext& context,
                                              const ov::Output<ov::Node>& complex_data,
                                              bool squeezed = true) {
        return get_complex_part_by_index(context, complex_data, 1, squeezed);
    }

    // Real and imaginary parts are broadcasted to each other and concatenated along
    // newly added dimensions in the tail
    static ov::Output<ov::Node> create_complex_tensor(const NodeContext& context,
                                                      const ov::Output<ov::Node>& real_part,
                                                      const ov::Output<ov::Node>& imag_part,
                                                      bool needs_broadcast = false);

    // Compute summation of two operands that can be of complex type
    // if operand is of complex type, complex type will be indicated by bool flag
    // complex tensor is represented as a real tensor with auxiliary dimension 2 in the tail
    // types of both operands must be aligned prior to the call
    static ov::Output<ov::Node> add(const NodeContext& context,
                                    const ov::Output<ov::Node>& lhs,
                                    const ov::Output<ov::Node>& rhs,
                                    bool lhs_complex = false,
                                    bool rhs_complex = false);

    // Compute subtraction of two operands that can be of complex type
    // if operand is of complex type, complex type will be indicated by bool flag
    // complex tensor is represented as a real tensor with auxiliary dimension 2 in the tail
    // types of both operands must be aligned prior to the call
    static ov::Output<ov::Node> sub(const NodeContext& context,
                                    const ov::Output<ov::Node>& lhs,
                                    const ov::Output<ov::Node>& rhs,
                                    bool lhs_complex = false,
                                    bool rhs_complex = false);

    // Compute multiplication of two operands that can be of complex type
    // if operand is of complex type, complex type will be indicated by bool flag
    // complex tensor is represented as a real tensor with auxiliary dimension 2 in the tail
    // types of both operands must be aligned prior to the call
    static ov::Output<ov::Node> mul(const NodeContext& context,
                                    const ov::Output<ov::Node>& lhs,
                                    const ov::Output<ov::Node>& rhs,
                                    bool lhs_complex = false,
                                    bool rhs_complex = false);

    // Compute inverse of operand that can be of complex type
    // if operand is of complex type, complex type will be indicated by bool flag
    // complex tensor is represented as a real tensor with auxiliary dimension 2 in the tail
    static ov::Output<ov::Node> inv(const NodeContext& context,
                                    const ov::Output<ov::Node>& data,
                                    bool data_complex = false);

    // Compute division of two operands that can be of complex type
    // if operand is of complex type, complex type will be indicated by bool flag
    // complex tensor is represented as a real tensor with auxiliary dimension 2 in the tail
    // types of both operands must be aligned prior to the call
    static ov::Output<ov::Node> div(const NodeContext& context,
                                    const ov::Output<ov::Node>& lhs,
                                    const ov::Output<ov::Node>& rhs,
                                    bool lhs_complex = false,
                                    bool rhs_complex = false);

private:
    static ov::Output<ov::Node> get_complex_part_by_index(const NodeContext& context,
                                                          const ov::Output<ov::Node>& complex_data,
                                                          int32_t index,
                                                          bool squeezed);

    ov::element::Type m_complex_part_type;
};

}  // namespace frontend
}  // namespace ov
