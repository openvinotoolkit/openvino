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
        NGRAPH_API
        NodeVector get_output_elements(const std::shared_ptr<Node>& mon);

        namespace v0
        {
            /// \brief Operation to get an output from a node.
            class NGRAPH_API GetOutputElement : public Op
            {
            public:
                static constexpr NodeTypeInfo type_info{"GetOutputElement", 0};
                const NodeTypeInfo& get_type_info() const override { return type_info; }
                GetOutputElement() = default;
                /// \brief Constructs a get-tuple-element operation.
                ///
                /// \param arg The input tuple.
                /// \param n The index of the tuple element to get.
                GetOutputElement(const std::shared_ptr<Node>& arg, size_t n);

                /// Return the equilent Output<Node>
                Output<Node> get_as_output() const;

                std::shared_ptr<Node>
                    clone_with_new_inputs(const OutputVector& inputs) const override;
                void validate_and_infer_types() override;

                /// \return The index of the tuple element to get.
                size_t get_n() const { return m_n; }
                NodeVector get_arguments() const override;

            protected:
                virtual void generate_adjoints(autodiff::Adjoints& adjoints,
                                               const OutputVector& deltas) override;
                size_t m_n;
            };
        }
        using v0::GetOutputElement;
    }

    inline std::shared_ptr<Node> get_output_element(const Output<Node>& output)
    {
        return output.get_node_shared_ptr()->get_output_as_single_output_node(output.get_index());
    }

    inline std::shared_ptr<Node> get_output_element(const std::shared_ptr<Node> node, size_t i = 0)
    {
        return node->get_output_as_single_output_node(i);
    }
}
