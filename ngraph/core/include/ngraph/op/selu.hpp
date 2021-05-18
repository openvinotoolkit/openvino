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
        namespace v0
        {
            /// \brief Performs a SELU activation function on all elements of the input node
            class NGRAPH_API Selu : public ngraph::op::Op
            {
            public:
                NGRAPH_RTTI_DECLARATION;

                Selu() = default;
                /// \brief Constructs a Selu node.
                ///
                /// \param data - Node producing the input tensor
                /// \param alpha - Alpha coefficient of SELU operation
                /// \param lambda - Lambda coefficient of SELU operation
                Selu(const Output<Node>& data,
                     const Output<Node>& alpha,
                     const Output<Node>& lambda);

                void validate_and_infer_types() override;

                bool visit_attributes(AttributeVisitor& visitor) override;

                virtual std::shared_ptr<Node>
                    clone_with_new_inputs(const OutputVector& new_args) const override;
            };
        } // namespace v0
        using v0::Selu;
    } // namespace op
} // namespace ngraph
