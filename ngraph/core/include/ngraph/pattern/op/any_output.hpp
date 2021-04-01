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
            /// Matches any output of a node
            class NGRAPH_API AnyOutput : public Pattern
            {
            public:
                static constexpr NodeTypeInfo type_info{"patternAnyOutput", 0};
                const NodeTypeInfo& get_type_info() const override;
                /// \brief creates an AnyOutput node matching any output of a node
                /// \param node The node to match
                AnyOutput(const std::shared_ptr<Node>& pattern)
                    : Pattern({pattern->output(0)})
                {
                }

                bool match_value(pattern::Matcher* matcher,
                                 const Output<Node>& pattern_value,
                                 const Output<Node>& graph_value) override;
            };
        }
    }
}
