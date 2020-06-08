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

#include "ngraph/op/util/binary_elementwise_arithmetic.hpp"

namespace ngraph
{
    namespace op
    {
        namespace v0
        {
            /// \brief Elementwise subtraction operation.
            class NGRAPH_API Subtract : public util::BinaryElementwiseArithmetic
            {
            public:
                static constexpr NodeTypeInfo type_info{"Subtract", 0};
                const NodeTypeInfo& get_type_info() const override { return type_info; }
                Subtract()
                    : util::BinaryElementwiseArithmetic(AutoBroadcastSpec::NONE)
                {
                }

                /// \brief Constructs a subtraction operation.
                ///
                /// \param arg0 Node that produces the first input tensor.
                /// \param arg1 Node that produces the second input tensor.
                /// \param auto_broadcast Auto broadcast specification
                Subtract(const Output<Node>& arg0,
                         const Output<Node>& arg1,
                         const AutoBroadcastSpec& auto_broadcast = AutoBroadcastSpec());

                virtual std::shared_ptr<Node>
                    clone_with_new_inputs(const OutputVector& new_args) const override;

                virtual void generate_adjoints(autodiff::Adjoints& adjoints,
                                               const OutputVector& deltas) override;

                bool evaluate(const HostTensorVector& outputs,
                              const HostTensorVector& inputs) override;
            };

        } // namespace v0

        namespace v1
        {
            /// \brief Elementwise subtraction operation.
            class NGRAPH_API Subtract : public util::BinaryElementwiseArithmetic
            {
            public:
                RTTI_DECLARATION;

                Subtract()
                    : util::BinaryElementwiseArithmetic(AutoBroadcastSpec::NUMPY)
                {
                }

                /// \brief Constructs a subtraction operation.
                ///
                /// \param arg0 Node that produces the first input tensor.
                /// \param arg1 Node that produces the second input tensor.
                /// \param auto_broadcast Auto broadcast specification
                Subtract(const Output<Node>& arg0,
                         const Output<Node>& arg1,
                         const AutoBroadcastSpec& auto_broadcast =
                             AutoBroadcastSpec(AutoBroadcastType::NUMPY));

                virtual std::shared_ptr<Node>
                    clone_with_new_inputs(const OutputVector& new_args) const override;

                virtual void generate_adjoints(autodiff::Adjoints& adjoints,
                                               const OutputVector& deltas) override;
                bool evaluate(const HostTensorVector& outputs,
                              const HostTensorVector& inputs) override;
            };
        } // namespace v1

        using v0::Subtract;
    } // namespace op

    NGRAPH_API
    std::shared_ptr<ngraph::Node> operator-(const Output<ngraph::Node> arg0,
                                            const Output<ngraph::Node> arg1);
} // namespace ngraph
