// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/op/op.hpp"
#include "ngraph/node.hpp"
#include "openvino/op/util/unary_elementwise_arithmetic.hpp"

namespace GNAPluginNS {
/// \brief Neural Activation Function
/// f(x) =  x/(1.0 + |x|)
///
class SoftSign : public ov::op::util::UnaryElementwiseArithmetic {
public:
    NGRAPH_RTTI_DECLARATION;

    SoftSign() = default;
    /// \brief Constructs an SoftSign operation.
    ///
    /// \param data Input tensor
    SoftSign(const ngraph::Output<ngraph::Node>& arg);
    std::shared_ptr<Node> clone_with_new_inputs(const ngraph::OutputVector& new_args) const override;
    OPENVINO_SUPPRESS_DEPRECATED_START
    bool evaluate(const ngraph::HostTensorVector& outputs, const ngraph::HostTensorVector& inputs) const override;
    OPENVINO_SUPPRESS_DEPRECATED_END
    bool has_evaluate() const override;
};
}  // namespace GNAPluginNS
