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
            /// \brief My operation.
            class NGRAPH_API MyNode : public util::UnaryElementwiseArithmetic
            {
            public:
                static constexpr NodeTypeInfo type_info{"MyNode", 0};
                const NodeTypeInfo& get_type_info() const override { return type_info; }
                /// \brief Constructs my operation.
                MyNode() = default;
                /// \brief Constructs my operation.
                ///
                /// \param arg Node that produces the input tensor.
                MyNode(const Output<Node>& arg);

                bool visit_attributes(AttributeVisitor& visitor) override;
                virtual std::shared_ptr<Node>
                clone_with_new_inputs(const OutputVector& new_args) const override;
                bool evaluate(const HostTensorVector& outputs,
                              const HostTensorVector& inputs) const override;
                bool has_evaluate() const override;
            };
        } // namespace v0
        using v0::MyNode;
    } // namespace op
} // namespace ngraph
