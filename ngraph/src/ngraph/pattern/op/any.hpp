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
            /// The graph value is to the matched value list. If the predicate is true for the node
            /// and the arguments match, the match succeeds.
            class NGRAPH_API Any : public Pattern
            {
            public:
                static constexpr NodeTypeInfo type_info{"patternAny", 0};
                const NodeTypeInfo& get_type_info() const override;
                /// \brief creates a Any node containing a sub-pattern described by \sa type and \sa
                ///        shape.
                Any(const element::Type& type,
                    const PartialShape& s,
                    ValuePredicate pred,
                    const OutputVector& wrapped_values)
                    : Pattern(wrapped_values, pred)
                {
                    set_output_type(0, type, s);
                }
                Any(const element::Type& type,
                    const PartialShape& s,
                    NodePredicate pred,
                    const NodeVector& wrapped_values)
                    : Any(type, s, as_value_predicate(pred), as_output_vector(wrapped_values))
                {
                }
                /// \brief creates a Any node containing a sub-pattern described by the type and
                ///        shape of \sa node.
                Any(const Output<Node>& node,
                    ValuePredicate pred,
                    const OutputVector& wrapped_values)
                    : Any(node.get_element_type(), node.get_partial_shape(), pred, wrapped_values)
                {
                }
                Any(const Output<Node>& node, NodePredicate pred, const NodeVector& wrapped_values)
                    : Any(node.get_element_type(),
                          node.get_partial_shape(),
                          as_value_predicate(pred),
                          as_output_vector(wrapped_values))
                {
                }

                bool match_value(pattern::Matcher* matcher,
                                 const Output<Node>& pattern_value,
                                 const Output<Node>& graph_value) override;
            };
        }
    }
}
