//*****************************************************************************
// Copyright 2017-2021 Intel Corporation
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
            /// A submatch on the graph value is performed on each input to the Or; the match
            /// succeeds on the first match. Otherwise the match fails.
            class NGRAPH_API Or : public Pattern
            {
            public:
                static constexpr NodeTypeInfo type_info{"patternOr", 0};
                const NodeTypeInfo& get_type_info() const override;
                /// \brief creates an Or node matching one of several sub-patterns in order. Does
                /// not add node to match list.
                /// \param patterns The patterns to try for matching
                Or(const OutputVector& patterns)
                    : Pattern(patterns)
                {
                }

                bool match_value(pattern::Matcher* matcher,
                                 const Output<Node>& pattern_value,
                                 const Output<Node>& graph_value) override;
            };
        }
    }
}
