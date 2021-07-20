// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>

#include "ngraph/op/util/unary_elementwise_arithmetic.hpp"

namespace ngraph
{
    namespace op
    {
        namespace v0
        {
            /// \brief Elementwise absolute value operation.
            ///
            class NGRAPH_API Abs : public util::UnaryElementwiseArithmetic
            {
            public:
                static constexpr NodeTypeInfo type_info{"Abs", 0};
                const NodeTypeInfo& get_type_info() const override { return type_info; }
                /// \brief Constructs an absolute value operation.
                Abs() = default;
                bool visit_attributes(AttributeVisitor& visitor) override
                {
                    (void)visitor;
                    return true;
                }
                /// \brief Constructs an absolute value operation.
                ///
                /// \param arg Output that produces the input tensor.<br>
                /// `[d1, ...]`
                ///
                /// Output `[d1, ...]`
                ///
                Abs(const Output<Node>& arg);

                std::shared_ptr<Node>
                    clone_with_new_inputs(const OutputVector& new_args) const override;

                bool evaluate(const HostTensorVector& outputs,
                              const HostTensorVector& inputs) const override;
                bool has_evaluate() const override;
            };
        } // namespace v0
        using v0::Abs;
    } // namespace op
} // namespace ngraph
