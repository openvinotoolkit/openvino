// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ngraph/node.hpp"
#include "ngraph/op/util/fused_op.hpp"

NGRAPH_SUPPRESS_DEPRECATED_START

namespace ngraph
{
    namespace op
    {
        namespace v0
        {
            /// \brief      Parameterized, bounded sigmoid-like, piecewise linear
            ///             function. min(max(alpha*x + beta, 0), 1)
            ///
            class NGRAPH_API HardSigmoid : public ngraph::op::util::FusedOp
            {
            public:
                static constexpr NodeTypeInfo type_info{"HardSigmoid", 0};
                const NodeTypeInfo& get_type_info() const override { return type_info; }
                HardSigmoid();

                /// \brief      Constructs a HardSigmoid operation.
                ///
                /// \param      data   Input tensor.
                /// \param[in]  alpha  A scalar value representing the alpha parameter.
                /// \param[in]  beta   A scalar value representing the beta parameter.
                ///
                HardSigmoid(const Output<Node>& data,
                            const Output<Node>& alpha,
                            const Output<Node>& beta);

                bool visit_attributes(AttributeVisitor& visitor) override;
                virtual void pre_validate_and_infer_types() override;
                virtual OutputVector decompose_op() const override;
                virtual std::shared_ptr<Node>
                    clone_with_new_inputs(const OutputVector& new_args) const override;
            };
        } // namespace v0
        using v0::HardSigmoid;
    } // namespace op
} // namespace ngraph

NGRAPH_SUPPRESS_DEPRECATED_END
