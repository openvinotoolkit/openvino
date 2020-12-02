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

#include "ngraph/node.hpp"
#include "ngraph/pattern/op/pattern.hpp"

namespace ngraph
{
    namespace pattern
    {
        namespace op
        {
            /// A branch adds a loop to the pattern. The branch match is successful if the
            /// destination node pattern matches the graph value. The destination node is a node in
            /// the pattern graph that will not have been created some time after the Branch node is
            /// created; use set_destination to add it.
            ///
            /// The branch destination is not stored as a shared pointer to prevent reference
            /// cycles. Thus the destination node must be referenced in some other way to prevent it
            /// from being deleted.
            class NGRAPH_API Branch : public Pattern
            {
            public:
                static constexpr NodeTypeInfo type_info{"patternBranch", 0};
                const NodeTypeInfo& get_type_info() const override;
                /// \brief Creates a Branch pattern
                /// \param pattern the destinationing pattern
                /// \param labels Labels where the destination may occur
                Branch()
                    : Pattern(OutputVector{})
                {
                    set_output_type(0, element::Type_t::f32, Shape{});
                }

                void set_destination(const Output<Node>& destination)
                {
                    m_destination_node = destination.get_node();
                    m_destination_index = destination.get_index();
                }

                Output<Node> get_destination() const
                {
                    return m_destination_node == nullptr
                               ? Output<Node>()
                               : Output<Node>{m_destination_node->shared_from_this(),
                                              m_destination_index};
                }

                bool match_value(pattern::Matcher* matcher,
                                 const Output<Node>& pattern_value,
                                 const Output<Node>& graph_value) override;

            protected:
                Node* m_destination_node{nullptr};
                size_t m_destination_index{0};
            };
        }
    }
}
