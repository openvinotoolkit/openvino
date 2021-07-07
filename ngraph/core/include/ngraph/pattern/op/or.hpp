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
        } // namespace op
    }     // namespace pattern
} // namespace ngraph
