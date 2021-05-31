// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>

#include "ngraph/op/util/binary_elementwise_logical.hpp"

namespace ngraph
{
    namespace op
    {
        namespace v1
        {
            /// \brief Elementwise logical-xor operation.
            ///
            class NGRAPH_API LogicalXor : public util::BinaryElementwiseLogical
            {
            public:
                NGRAPH_RTTI_DECLARATION;
                LogicalXor() = default;
                /// \brief Constructs a logical-xor operation.
                ///
                /// \param arg0 Node that produces the first input tensor.<br>
                /// `[d0, ...]`
                /// \param arg1 Node that produces the second input tensor.<br>
                /// `[d0, ...]`
                /// \param auto_broadcast Auto broadcast specification
                ///
                /// Output `[d0, ...]`
                ///
                LogicalXor(const Output<Node>& arg0,
                           const Output<Node>& arg1,
                           const AutoBroadcastSpec& auto_broadcast =
                               AutoBroadcastSpec(AutoBroadcastType::NUMPY));

                virtual std::shared_ptr<Node>
                    clone_with_new_inputs(const OutputVector& new_args) const override;

                bool visit_attributes(AttributeVisitor& visitor) override;
                bool evaluate(const HostTensorVector& outputs,
                              const HostTensorVector& inputs) const override;
                bool has_evaluate() const override;
            };
        } // namespace v1
        namespace v0
        {
            /// \brief Elementwise logical-xor operation.
            ///
            class NGRAPH_API Xor : public util::BinaryElementwiseLogical
            {
            public:
                static constexpr NodeTypeInfo type_info{"Xor", 0};
                const NodeTypeInfo& get_type_info() const override { return type_info; }
                Xor() = default;
                /// \brief Constructs a logical-xor operation.
                ///
                /// \param arg0 Node that produces the first input tensor.<br>
                /// `[d0, ...]`
                /// \param arg1 Node that produces the second input tensor.<br>
                /// `[d0, ...]`
                /// \param auto_broadcast Auto broadcast specification
                ///
                /// Output `[d0, ...]`
                ///
                Xor(const Output<Node>& arg0,
                    const Output<Node>& arg1,
                    const AutoBroadcastSpec& auto_broadcast = AutoBroadcastSpec());

                virtual std::shared_ptr<Node>
                    clone_with_new_inputs(const OutputVector& new_args) const override;

                bool evaluate(const HostTensorVector& outputs,
                              const HostTensorVector& inputs) const override;
                bool has_evaluate() const override;
            };
        } // namespace v0

        // default opset version
        using v0::Xor;
    } // namespace op
} // namespace ngraph
