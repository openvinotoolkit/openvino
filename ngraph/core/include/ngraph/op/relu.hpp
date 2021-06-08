// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ngraph/node.hpp"
#include "ngraph/op/op.hpp"
#include "ngraph/op/util/binary_elementwise_arithmetic.hpp"
#include "ngraph/op/util/unary_elementwise_arithmetic.hpp"

#include <memory>

namespace ngraph
{
    namespace op
    {
        namespace v0
        {
            /// \brief Elementwise Relu operation.
            ///
            class NGRAPH_API Relu : public ngraph::op::util::UnaryElementwiseArithmetic
            {
            public:
                NGRAPH_RTTI_DECLARATION;
                Relu() = default;
                /// \brief Constructs a Relu operation.
                ///
                /// \param arg Node that produces the input tensor.
                Relu(const Output<ngraph::Node>& arg);

                virtual std::shared_ptr<Node>
                    clone_with_new_inputs(const OutputVector& new_args) const override;

                bool evaluate(const HostTensorVector& outputs,
                              const HostTensorVector& inputs) const override;
                bool has_evaluate() const override;
                bool visit_attributes(AttributeVisitor& visitor) override;
            };
        } // namespace v0
        using v0::Relu;
    } // namespace op
} // namespace ngraph
