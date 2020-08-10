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
#include "ngraph/runtime/host_tensor.hpp"

namespace ngraph
{
    namespace op
    {
        namespace v0
        {
            /// \brief Elementwise type conversion operation.
            class NGRAPH_API Convert : public Op
            {
            public:
                NGRAPH_RTTI_DECLARATION;

                /// \brief Constructs a conversion operation.
                Convert() = default;
                /// \brief Constructs a conversion operation.
                ///
                /// \param arg          Node that produces the input tensor.
                /// \param destination_type  Element type for the output tensor.
                Convert(const Output<Node>& arg, const ngraph::element::Type& destination_type);

                void validate_and_infer_types() override;
                bool visit_attributes(AttributeVisitor& visitor) override;
                virtual std::shared_ptr<Node>
                    clone_with_new_inputs(const OutputVector& new_args) const override;
                const element::Type& get_destination_type() const { return m_destination_type; }
                void set_destination_type(const element::Type& destination_type)
                {
                    m_destination_type = destination_type;
                }
                const element::Type& get_convert_element_type() const { return m_destination_type; }
                void set_convert_element_type(const element::Type& destination_type)
                {
                    m_destination_type = destination_type;
                }

                bool evaluate(const HostTensorVector& outputs,
                              const HostTensorVector& inputs) const override;

            protected:
                ngraph::element::Type m_destination_type;
            };
        }
        using v0::Convert;
    }
}
