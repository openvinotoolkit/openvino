// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ngraph/node.hpp"
#include "ngraph/op/op.hpp"
#include "ngraph/op/util/unary_elementwise_arithmetic.hpp"

namespace ngraph
{
    namespace op
    {
        namespace v4
        {
            /// \brief A HSwish Activation Function
            /// f(x) =  x * min(max(x + 3, 0), 6) / 6 or
            /// f(x) = x * min(ReLU(x + 3), 6) / 6
            ///
            class NGRAPH_API HSwish : public ngraph::op::util::UnaryElementwiseArithmetic
            {
            public:
                NGRAPH_RTTI_DECLARATION;
                HSwish() = default;

                /// \brief Constructs a HSwish (hard version of Swish) operation.
                ///
                /// \param data Input tensor
                HSwish(const Output<Node>& arg);

                bool visit_attributes(AttributeVisitor& visitor) override;

                virtual std::shared_ptr<Node>
                    clone_with_new_inputs(const OutputVector& new_args) const override;
                bool evaluate(const HostTensorVector& outputs,
                              const HostTensorVector& inputs) const override;
                bool has_evaluate() const override;
            };
        } // namespace v4
    }     // namespace op
} // namespace ngraph
