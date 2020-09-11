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
            /// The graph value is added to the matched value list. If the predicate is true, the
            /// match succeeds if the arguments match; if the predicate is false, the match succeeds
            /// if the pattern input matches the graph value.
            class NGRAPH_API Skip : public Pattern
            {
            public:
                static constexpr NodeTypeInfo type_info{"patternSkip", 0};
                const NodeTypeInfo& get_type_info() const override;
                Skip(const Output<Node>& arg, ValuePredicate pred)
                    : Pattern({arg}, pred)
                {
                    set_output_type(0, arg.get_element_type(), arg.get_partial_shape());
                }

                Skip(const Output<Node>& arg, NodePredicate pred = nullptr)
                    : Pattern({arg}, as_value_predicate(pred))
                {
                    set_output_type(0, arg.get_element_type(), arg.get_partial_shape());
                }

                virtual bool match_value(pattern::Matcher* matcher,
                                         const Output<Node>& pattern_value,
                                         const Output<Node>& graph_value) override;
            };
        }
    }
}
