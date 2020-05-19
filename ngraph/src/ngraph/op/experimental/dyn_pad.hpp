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
            /// \brief Generic padding operation which takes padding below and above as dynamic
            /// shapes.
            /// This is similar to existing Pad operation except padding values are dynamic.
            class NGRAPH_API DynPad : public Op
            {
            public:
                static constexpr NodeTypeInfo type_info{"DynPad", 0};
                const NodeTypeInfo& get_type_info() const override { return type_info; }
                DynPad() = default;
                /// \brief Perform dynamic padding of a tensor
                ///
                /// \param arg The node producing input tensor to be padded.
                /// \param padding_below The node producing the padding-below widths.
                /// \param padding_above The node producing the padding-above widths.
                /// \param padding_value The value to be used for padding. Must be scalar.
                /// \param pad_mode The padding mode: CONSTANT(default), EDGE or REFLECT.
                DynPad(const Output<Node>& arg,
                       const Output<Node>& padding_below,
                       const Output<Node>& padding_above,
                       const Output<Node>& padding_value,
                       PadMode pad_mode = PadMode::CONSTANT);

                PadMode get_pad_mode() const { return m_pad_mode; }
                void validate_and_infer_types() override;

                virtual std::shared_ptr<Node>
                    clone_with_new_inputs(const OutputVector& new_args) const override;

            protected:
                virtual void generate_adjoints(autodiff::Adjoints& adjoints,
                                               const OutputVector& deltas) override;

            private:
                PadMode m_pad_mode;
            };
        }
        using v0::DynPad;
    }
}
