// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ngraph/node.hpp"
#include "ngraph/op/op.hpp"
#include "ngraph/op/util/fused_op.hpp"

NGRAPH_SUPPRESS_DEPRECATED_START

namespace ngraph
{
    namespace op
    {
        namespace v0
        {
            /// \brief Performs a SELU activation function on all elements of the input node
            class NGRAPH_API Selu : public ngraph::op::util::FusedOp
            {
            public:
                static constexpr NodeTypeInfo type_info{"Selu", 0};
                const NodeTypeInfo& get_type_info() const override { return type_info; }
                Selu();
                /// \brief Constructs a Selu node.
                ///
                /// \param data - Node producing the input tensor
                /// \param alpha - Alpha coefficient of SELU operation
                /// \param lambda - Lambda coefficient of SELU operation
                Selu(const Output<Node>& data,
                     const Output<Node>& alpha,
                     const Output<Node>& lambda);
                virtual void pre_validate_and_infer_types() override;
                bool visit_attributes(AttributeVisitor& visitor) override;
                virtual OutputVector decompose_op() const override;

                virtual std::shared_ptr<Node>
                    clone_with_new_inputs(const OutputVector& new_args) const override;
            };
        } // namespace v0
        using v0::Selu;
    } // namespace op
} // namespace ngraph

NGRAPH_SUPPRESS_DEPRECATED_END
