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
            /// \brief Elementwise cosine operation.
            class NGRAPH_API Cos : public util::UnaryElementwiseArithmetic
            {
            public:
                static constexpr NodeTypeInfo type_info{"Cos", 0};
                const NodeTypeInfo& get_type_info() const override { return type_info; }
                /// \brief Constructs a cosine operation.
                Cos() = default;
                /// \brief Constructs a cosine operation.
                ///
                /// \param arg Node that produces the input tensor.
                Cos(const Output<Node>& arg);
                bool visit_attributes(AttributeVisitor& visitor) override;

                virtual std::shared_ptr<Node>
                    clone_with_new_inputs(const OutputVector& new_args) const override;
                bool evaluate(const HostTensorVector& outputs,
                              const HostTensorVector& inputs) const override;
                bool has_evaluate() const override;
            };
        } // namespace v0
        using v0::Cos;
    } // namespace op
} // namespace ngraph
