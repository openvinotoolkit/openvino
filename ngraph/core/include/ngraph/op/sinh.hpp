// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ngraph/op/util/unary_elementwise_arithmetic.hpp"

namespace ngraph
{
    namespace op
    {
        namespace v0
        {
            /// \brief Elementwise hyperbolic sine (sinh) operation.
            class NGRAPH_API Sinh : public util::UnaryElementwiseArithmetic
            {
            public:
                static constexpr NodeTypeInfo type_info{"Sinh", 0};
                const NodeTypeInfo& get_type_info() const override { return type_info; }
                /// \brief Constructs a hyperbolic sine operation.
                ///
                /// \param arg Node that produces the input tensor.
                Sinh(const Output<Node>& arg);
                Sinh() = default;

                bool visit_attributes(AttributeVisitor& visitor) override;
                virtual std::shared_ptr<Node>
                    clone_with_new_inputs(const OutputVector& new_args) const override;
                bool evaluate(const HostTensorVector& outputs,
                              const HostTensorVector& inputs) const override;
                bool has_evaluate() const override;
            };
        } // namespace v0
        using v0::Sinh;
    } // namespace op
} // namespace ngraph
