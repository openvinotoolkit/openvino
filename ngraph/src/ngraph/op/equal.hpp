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
            // clang-format off
            /// \brief Elementwise is-equal operation.
            ///
            /// ## Inputs
            ///
            /// |        | Type                              | Description                                            |
            /// | ------ | --------------------------------- | ------------------------------------------------------ |
            /// | `arg0` | \f$E[d_1,\dots,d_n]~(n \geq 0)\f$ | A tensor of any shape and element type.                |
            /// | `arg1` | \f$E[d_1,\dots,d_n]~(n \geq 0)\f$ | A tensor of the same shape and element type as `arg0`. |
            /// | `autob`| AutoBroadcastSpec                 | Auto broadcast specification.                          |
            ///
            /// ## Output
            ///
            /// | Type                               | Description                                                                                                                                |
            /// | ---------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------ |
            /// | \f$\texttt{bool}[d_1,\dots,d_n]\f$ | The tensor \f$T\f$, where \f$T[i_1,\dots,i_n] = 1\text{ if }\texttt{arg0}[i_1,\dots,i_n] = \texttt{arg1}[i_1,\dots,i_n]\text{, else } 0\f$ |
            // clang-format on
            class NGRAPH_API Equal : public util::BinaryElementwiseComparison
            {
            public:
                static constexpr NodeTypeInfo type_info{"Equal", 0};
                const NodeTypeInfo& get_type_info() const override { return type_info; }
                /// \brief Constructs an equal operation.
                Equal() = default;
                /// \brief Constructs an equal operation.
                ///
                /// \param arg0 Node that produces the first input tensor.
                /// \param arg1 Node that produces the second input tensor.
                /// \param auto_broadcast Auto broadcast specification
                Equal(const Output<Node>& arg0,
                      const Output<Node>& arg1,
                      const AutoBroadcastSpec& auto_broadcast = AutoBroadcastSpec());

                virtual std::shared_ptr<Node>
                    clone_with_new_inputs(const OutputVector& new_args) const override;

                virtual bool is_commutative() const override { return true; }
#ifdef NGRAPH_EVALUATE_ENABLE
                bool evaluate(const HostTensorVector& outputs,
                              const HostTensorVector& inputs) override;
#endif
            };
        } // namespace v0

        namespace v1
        {
            // clang-format off
            /// \brief Elementwise is-equal operation.
            ///
            /// ## Inputs
            ///
            /// |        | Type                              | Description                                            |
            /// | ------ | --------------------------------- | ------------------------------------------------------ |
            /// | `arg0` | \f$E[d_1,\dots,d_n]~(n \geq 0)\f$ | A tensor of any shape and element type.                |
            /// | `arg1` | \f$E[d_1,\dots,d_n]~(n \geq 0)\f$ | A tensor of the same shape and element type as `arg0`. |
            /// | `autob`| AutoBroadcastSpec                 | Auto broadcast specification.                          |
            ///
            /// ## Output
            ///
            /// | Type                               | Description                                                                                                                                |
            /// | ---------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------ |
            /// | \f$\texttt{bool}[d_1,\dots,d_n]\f$ | The tensor \f$T\f$, where \f$T[i_1,\dots,i_n] = 1\text{ if }\texttt{arg0}[i_1,\dots,i_n] = \texttt{arg1}[i_1,\dots,i_n]\text{, else } 0\f$ |
            // clang-format on
            class NGRAPH_API Equal : public util::BinaryElementwiseComparison
            {
            public:
                static constexpr NodeTypeInfo type_info{"Equal", 1};
                const NodeTypeInfo& get_type_info() const override { return type_info; }
                /// \brief Constructs an equal operation.
                Equal() = default;
                /// \brief Constructs an equal operation.
                ///
                /// \param arg0 Node that produces the first input tensor.
                /// \param arg1 Node that produces the second input tensor.
                /// \param auto_broadcast Auto broadcast specification
                Equal(const Output<Node>& arg0,
                      const Output<Node>& arg1,
                      const AutoBroadcastSpec& auto_broadcast =
                          AutoBroadcastSpec(AutoBroadcastType::NUMPY));

                virtual std::shared_ptr<Node>
                    clone_with_new_inputs(const OutputVector& new_args) const override;

                virtual bool is_commutative() const override { return true; }
#ifdef NGRAPH_EVALUATE_ENABLE
                bool evaluate(const HostTensorVector& outputs,
                              const HostTensorVector& inputs) override;
#endif
            };
        } // namespace v1

        using v0::Equal;
    }
}
