// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

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
        } // namespace op
    }     // namespace pattern
} // namespace ngraph
