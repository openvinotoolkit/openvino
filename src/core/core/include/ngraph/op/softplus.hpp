// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ngraph/node.hpp"
#include "ngraph/op/op.hpp"

namespace ngraph
{
    namespace op
    {
        namespace v4
        {
            /// \brief A Self Regularized Non-Monotonic Neural Activation Function
            /// f(x) =  ln(exp(x) + 1.)
            ///
            class NGRAPH_API SoftPlus : public ngraph::op::Op
            {
            public:
                NGRAPH_RTTI_DECLARATION;

                SoftPlus() = default;
                /// \brief Constructs an SoftPlus operation.
                ///
                /// \param data Input tensor
                SoftPlus(const Output<Node>& arg);
                bool visit_attributes(AttributeVisitor& visitor) override;
                void validate_and_infer_types() override;

                virtual std::shared_ptr<Node>
                    clone_with_new_inputs(const OutputVector& new_args) const override;

                bool evaluate(const HostTensorVector& outputs,
                              const HostTensorVector& inputs) const override;
                bool has_evaluate() const override;
            };
        } // namespace v4
    }     // namespace op
} // namespace ngraph
