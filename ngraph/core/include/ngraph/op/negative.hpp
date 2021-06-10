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
            /// \brief Elementwise negative operation.
            class NGRAPH_API Negative : public util::UnaryElementwiseArithmetic
            {
            public:
                static constexpr NodeTypeInfo type_info{"Negative", 0};
                const NodeTypeInfo& get_type_info() const override { return type_info; }
                /// \brief Constructs a negative operation.
                Negative() = default;
                /// \brief Constructs a negative operation.
                ///
                /// \param arg Node that produces the input tensor.
                Negative(const Output<Node>& arg);

                bool visit_attributes(AttributeVisitor& visitor) override;
                virtual std::shared_ptr<Node>
                    clone_with_new_inputs(const OutputVector& new_args) const override;
                bool evaluate(const HostTensorVector& outputs,
                              const HostTensorVector& inputs) const override;
                bool has_evaluate() const override;
            };
        } // namespace v0
        using v0::Negative;
    } // namespace op
    NGRAPH_API
    std::shared_ptr<Node> operator-(const Output<Node>& arg0);
} // namespace ngraph
