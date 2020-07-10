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

#include "ngraph/op/op.hpp"

namespace ngraph
{
    namespace op
    {
        namespace v0
        {
            // clang-format off
            /// \brief Elementwise selection operation.
            ///
            /// ## Inputs
            ///
            /// |        | Type                                          | Description                                                  |
            /// | ------ | --------------------------------------------- | ------------------------------------------------------------ |
            /// | `arg0` | \f$\texttt{bool}[d_1,\dots,d_n]~(n \geq 0)\f$ | A tensor of any shape, with element `bool`.                  |
            /// | `arg1` | \f$E[d_1,\dots,d_n]~(n \geq 0)\f$             | A tensor of the same shape as `arg0`, with any element type. |
            /// | `arg2` | \f$E[d_1,\dots,d_n]~(n \geq 0)\f$             | A tensor of the same shape and element type as `arg1`.       |
            ///
            /// ## Output
            ///
            /// | Type                   | Description                                                                                                                                                             |
            /// | ---------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
            /// | \f$E[d_1,\dots,d_n]\f$ | The tensor \f$T\f$, where \f$T[i_1,\dots,i_n] = \texttt{arg1}[i_1,\dots,i_n]\text{ if }\texttt{arg0}[i_1,\dots,i_n] \neq 0\text{, else }\texttt{arg2}[i_1,\dots,i_n]\f$ |
            // clang-format on
            class NGRAPH_API Select : public Op
            {
            public:
                static constexpr NodeTypeInfo type_info{"Select", 0};
                const NodeTypeInfo& get_type_info() const override { return type_info; }
                /// \brief Constructs a selection operation.
                Select() = default;
                /// \brief Constructs a selection operation.
                ///
                /// \param arg0 Node that produces the first input tensor.
                /// \param arg1 Node that produces the second input tensor.
                /// \param arg2 Node that produces the third input tensor.
                Select(const Output<Node>& arg0,
                       const Output<Node>& arg1,
                       const Output<Node>& arg2);

                virtual std::shared_ptr<Node>
                    clone_with_new_inputs(const OutputVector& new_args) const override;
                void validate_and_infer_types() override;
            };
        }

        namespace v1
        {
            // clang-format off
            /// \brief Elementwise selection operation.
            ///
            /// ## Inputs
            ///
            /// |        | Type                                          | Description                                                  |
            /// | ------ | --------------------------------------------- | ------------------------------------------------------------ |
            /// | `arg0` | \f$\texttt{bool}[d_1,\dots,d_n]~(n \geq 0)\f$ | A tensor of any shape, with element `bool`.                  |
            /// | `arg1` | \f$E[d_1,\dots,d_n]~(n \geq 0)\f$             | A tensor of a shape that is broadcast-compatible with `arg0`, with any element type. |
            /// | `arg2` | \f$E[d_1,\dots,d_n]~(n \geq 0)\f$             | A tensor of a shape that is broadcast-compatible with `arg0`, and same element type as `arg1`. |
            /// | `auto_broadcast`| AutoBroadcastSpec                             | Auto broadcast specification.                                |
            ///
            /// ## Output
            ///
            /// | Type                   | Description                                                                                                                                                             |
            /// | ---------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
            /// | \f$E[d_1,\dots,d_n]\f$ | The tensor \f$T\f$, where \f$T[i_1,\dots,i_n] = \texttt{arg1}[i_1,\dots,i_n]\text{ if }\texttt{arg0}[i_1,\dots,i_n] \neq 0\text{, else }\texttt{arg2}[i_1,\dots,i_n]\f$ |
            // clang-format on
            class NGRAPH_API Select : public Op
            {
            public:
                static constexpr NodeTypeInfo type_info{"Select", 1};
                const NodeTypeInfo& get_type_info() const override { return type_info; }
                /// \brief Constructs a selection operation.
                Select()
                    : m_auto_broadcast(AutoBroadcastSpec(AutoBroadcastType::NUMPY))
                {
                }

                /// \brief Constructs a selection operation.
                ///
                /// \param arg0 Node that produces the first input tensor.
                /// \param arg1 Node that produces the second input tensor.
                /// \param arg2 Node that produces the third input tensor.
                /// \param auto_broadcast Auto broadcast specification. Default is Numpy-style
                ///                       implicit broadcasting.
                Select(const Output<Node>& arg0,
                       const Output<Node>& arg1,
                       const Output<Node>& arg2,
                       const AutoBroadcastSpec& auto_broadcast =
                           AutoBroadcastSpec(AutoBroadcastType::NUMPY));

                virtual std::shared_ptr<Node>
                    clone_with_new_inputs(const OutputVector& new_args) const override;
                bool visit_attributes(AttributeVisitor& visitor) override;
                void validate_and_infer_types() override;

                const AutoBroadcastSpec& get_auto_broadcast() const { return m_auto_broadcast; }
                void set_auto_broadcast(const AutoBroadcastSpec& auto_broadcast)
                {
                    m_auto_broadcast = auto_broadcast;
                }
                bool supports_auto_broadcast() const override { return true; }
                // TODO: Move all uses of get_autob to get_auto_broadcast() and remove this.
                const AutoBroadcastSpec& get_autob() const override { return m_auto_broadcast; }
            private:
                AutoBroadcastSpec m_auto_broadcast;
            };
        }
        using v0::Select;
    }
}
