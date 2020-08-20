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

#include "ngraph/op/util/binary_elementwise_comparison.hpp"

namespace ngraph
{
    namespace op
    {
        namespace v0
        {
            /// \brief Elementwise greater-than-or-equal operation.
            class NGRAPH_DEPRECATED(
                "This operation is deprecated and will be removed soon. "
                "Use v1::GreaterEqual instead of it.") NGRAPH_API GreaterEq
                : public util::BinaryElementwiseComparison
            {
                NGRAPH_SUPPRESS_DEPRECATED_START
            public:
                static constexpr NodeTypeInfo type_info{"GreaterEq", 0};
                const NodeTypeInfo& get_type_info() const override { return type_info; }
                /// \brief Constructs a greater-than-or-equal operation.
                GreaterEq()
                    : util::BinaryElementwiseComparison(AutoBroadcastSpec::NONE)
                {
                }
                /// \brief Constructs a greater-than-or-equal operation.
                ///
                /// \param arg0 Node that produces the first input tensor.
                /// \param arg1 Node that produces the second input tensor.
                /// \param auto_broadcast Auto broadcast specification
                GreaterEq(const Output<Node>& arg0,
                          const Output<Node>& arg1,
                          const AutoBroadcastSpec& auto_broadcast = AutoBroadcastSpec());

                virtual std::shared_ptr<Node>
                    clone_with_new_inputs(const OutputVector& new_args) const override;
                bool evaluate(const HostTensorVector& outputs,
                              const HostTensorVector& inputs) const override;
                NGRAPH_SUPPRESS_DEPRECATED_END
            };
        } // namespace v0

        namespace v1
        {
            /// \brief Elementwise greater-than-or-equal operation.
            class NGRAPH_API GreaterEqual : public util::BinaryElementwiseComparison
            {
            public:
                NGRAPH_RTTI_DECLARATION;
                /// \brief Constructs a greater-than-or-equal operation.
                GreaterEqual()
                    : util::BinaryElementwiseComparison(AutoBroadcastSpec::NUMPY)
                {
                }
                /// \brief Constructs a greater-than-or-equal operation.
                ///
                /// \param arg0 Node that produces the first input tensor.
                /// \param arg1 Node that produces the second input tensor.
                /// \param auto_broadcast Auto broadcast specification
                GreaterEqual(const Output<Node>& arg0,
                             const Output<Node>& arg1,
                             const AutoBroadcastSpec& auto_broadcast =
                                 AutoBroadcastSpec(AutoBroadcastType::NUMPY));

                virtual std::shared_ptr<Node>
                    clone_with_new_inputs(const OutputVector& new_args) const override;
                bool evaluate(const HostTensorVector& outputs,
                              const HostTensorVector& inputs) const override;
            };
        } // namespace v1

        NGRAPH_SUPPRESS_DEPRECATED_START
        using v0::GreaterEq;
        NGRAPH_SUPPRESS_DEPRECATED_END
    }
}
