// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/op/op.hpp"
#include "openvino/op/util/unary_elementwise_arithmetic.hpp"

namespace ov {
namespace op {
namespace v0 {
/// \brief Gaussian Error Linear Unit
/// f(x) = 0.5 * x * (1 + erf( x / sqrt(2) )
/// \ingroup ov_ops_cpp_api
class OPENVINO_API Gelu : public Op {
public:
    OPENVINO_OP("Gelu", "opset2", op::Op, 0);
    BWDCMP_RTTI_DECLARATION;

    Gelu();
    /// \brief Constructs a Gelu operation.
    ///
    /// \param data Input tensor
    Gelu(const Output<Node>& data);

    bool visit_attributes(AttributeVisitor& visitor) override;

    void validate_and_infer_types() override;

    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;
};
}  // namespace v0

/// \brief Specifies the approximation to calculate Gelu
enum class GeluApproximationMode { TANH, ERF };
OPENVINO_API std::ostream& operator<<(std::ostream& s, const GeluApproximationMode& type);

namespace v7 {
/// \brief Gaussian Error Linear Unit
/// f(x) = 0.5 * x * (1 + erf( x / sqrt(2) ) for "approximation" = "erf"
/// f(x) = 0.5 * x * (1 + tanh([sqrt(2 / pi)] * [x + 0.044715^3]) for "approximation" =
/// "tanh"
/// \ingroup ov_ops_cpp_api
class OPENVINO_API Gelu : public util::UnaryElementwiseArithmetic {
public:
    OPENVINO_OP("Gelu", "opset7", op::Op, 7);
    BWDCMP_RTTI_DECLARATION;

    Gelu() = default;
    /// \brief Constructs a Gelu operation.
    ///
    /// \param data Input tensor
    /// \param mode Approximation mode
    Gelu(const Output<Node>& data, GeluApproximationMode mode = GeluApproximationMode::ERF);

    bool visit_attributes(AttributeVisitor& visitor) override;

    void validate_and_infer_types() override;

    OPENVINO_SUPPRESS_DEPRECATED_START
    bool evaluate(const HostTensorVector& outputs, const HostTensorVector& inputs) const override;
    OPENVINO_SUPPRESS_DEPRECATED_END
    bool has_evaluate() const override;

    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;

    GeluApproximationMode get_approximation_mode() const;

private:
    GeluApproximationMode m_approximation_mode = GeluApproximationMode::ERF;
};
}  // namespace v7
}  // namespace op

template <>
class OPENVINO_API AttributeAdapter<op::GeluApproximationMode>
    : public EnumAttributeAdapterBase<op::GeluApproximationMode> {
public:
    AttributeAdapter(op::GeluApproximationMode& value) : EnumAttributeAdapterBase<op::GeluApproximationMode>(value) {}

    OPENVINO_RTTI("AttributeAdapter<ov::op::GeluApproximationMode>");
    BWDCMP_RTTI_DECLARATION;
};
}  // namespace ov
