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
        namespace v5
        {
            /// \brief A HSigmoid Activation Function
            /// f(x) = min(max(x + 3, 0), 6) / 6 or
            /// f(x) = min(ReLU(x + 3), 6) / 6
            ///
            class NGRAPH_API HSigmoid : public ngraph::op::util::UnaryElementwiseArithmetic
            {
            public:
                NGRAPH_RTTI_DECLARATION;
                HSigmoid() = default;

                /// \brief Constructs a HSigmoid operation.
                ///
                /// \param data Input tensor
                HSigmoid(const Output<Node>& arg);

                bool visit_attributes(AttributeVisitor& visitor) override;

                virtual std::shared_ptr<Node>
                    clone_with_new_inputs(const OutputVector& new_args) const override;
                bool evaluate(const HostTensorVector& outputs,
                              const HostTensorVector& inputs) const override;
                bool has_evaluate() const override;
            };
        } // namespace v5
    }     // namespace op
} // namespace ngraph
