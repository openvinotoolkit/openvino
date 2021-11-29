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
            /// \brief A Swish Activation Function
            /// f(x) =  x / (1.0 + exp(-beta * x)) or
            /// f(x) = x * sigmoid(beta * x)
            ///
            class NGRAPH_API Swish : public ngraph::op::Op
            {
            public:
                static constexpr NodeTypeInfo type_info{"Swish", 4};
                const NodeTypeInfo& get_type_info() const override { return type_info; }
                Swish() = default;

                /// \brief Constructs an Swish operation.
                ///
                /// \param data Input tensor
                /// \param beta Scalar with beta value. If the argument is not specified then use
                /// the default value 1.0
                Swish(const Output<Node>& arg, const Output<Node>& beta);
                explicit Swish(const Output<Node>& arg);

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
