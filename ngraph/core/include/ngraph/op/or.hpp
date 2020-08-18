//*****************************************************************************
// Copyright 2017-2020 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//*****************************************************************************

#pragma once

#include <memory>

#include "ngraph/op/util/binary_elementwise_logical.hpp"

namespace ngraph
{
    namespace op
    {
        namespace v1
        {
            /// \brief Elementwise logical-or operation.
            ///
            class NGRAPH_API LogicalOr : public util::BinaryElementwiseLogical
            {
            public:
                NGRAPH_RTTI_DECLARATION;
                LogicalOr() = default;
                /// \brief Constructs a logical-or operation.
                ///
                /// \param arg0 Node that produces the first input tensor.<br>
                /// `[d0, ...]`
                /// \param arg1 Node that produces the second input tensor.<br>
                /// `[d0, ...]`
                /// \param auto_broadcast Auto broadcast specification
                ///
                /// Output `[d0, ...]`
                ///
                LogicalOr(const Output<Node>& arg0,
                          const Output<Node>& arg1,
                          const AutoBroadcastSpec& auto_broadcast =
                              AutoBroadcastSpec(AutoBroadcastType::NUMPY));

                virtual std::shared_ptr<Node>
                    clone_with_new_inputs(const OutputVector& new_args) const override;

                bool evaluate(const HostTensorVector& outputs,
                              const HostTensorVector& inputs) const override;
            };
        } // namespace v1
        namespace v0
        {
            /// \brief Elementwise logical-or operation.
            ///
            class NGRAPH_API Or : public util::BinaryElementwiseLogical
            {
            public:
                static constexpr NodeTypeInfo type_info{"Or", 0};
                const NodeTypeInfo& get_type_info() const override { return type_info; }
                Or() = default;
                /// \brief Constructs a logical-or operation.
                ///
                /// \param arg0 Node that produces the first input tensor.<br>
                /// `[d0, ...]`
                /// \param arg1 Node that produces the second input tensor.<br>
                /// `[d0, ...]`
                /// \param auto_broadcast Auto broadcast specification
                ///
                /// Output `[d0, ...]`
                ///
                Or(const Output<Node>& arg0,
                   const Output<Node>& arg1,
                   const AutoBroadcastSpec& auto_broadcast = AutoBroadcastSpec());

                virtual std::shared_ptr<Node>
                    clone_with_new_inputs(const OutputVector& new_args) const override;

                bool evaluate(const HostTensorVector& outputs,
                              const HostTensorVector& inputs) const override;
            };
        } // namespace v0

        using v0::Or;
    } // namespace op
} // namespace ngraph
