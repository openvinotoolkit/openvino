// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/op/op.hpp"
#include "ngraph/node.hpp"
#include "openvino/op/util/unary_elementwise_arithmetic.hpp"

namespace ov {
namespace intel_gna {
namespace op {
/// \brief Neural Activation Function
/// f(x) =  x/(1.0 + |x|)
///
class SoftSign : public ov::op::util::UnaryElementwiseArithmetic {
public:
    OPENVINO_OP("SoftSign", "intel_gna", ov::op::util::UnaryElementwiseArithmetic);

    SoftSign() = default;
    /// \brief Constructs an SoftSign operation.
    ///
    /// \param data Input tensor
    SoftSign(const ngraph::Output<ngraph::Node>& arg);
    std::shared_ptr<Node> clone_with_new_inputs(const ngraph::OutputVector& new_args) const override;
    bool evaluate(ov::TensorVector& output_values,
                  const ov::TensorVector& input_values,
                  const ov::EvaluationContext & evaluation_context) const override;
    bool has_evaluate() const override;
};
} // namespace op
} // namespace intel_gna
} // namespace ov
