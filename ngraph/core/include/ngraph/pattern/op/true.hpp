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
            /// \brief The match always succeeds.
            class NGRAPH_API True : public Pattern
            {
            public:
                static constexpr NodeTypeInfo type_info{"patternTrue", 0};
                const NodeTypeInfo& get_type_info() const override;
                /// \brief Always matches, does not add node to match list.
                True()
                    : Pattern(OutputVector{})
                {
                }
                bool match_value(pattern::Matcher* matcher,
                                 const Output<Node>& pattern_value,
                                 const Output<Node>& graph_value) override;
            };
        } // namespace op
    }     // namespace pattern
} // namespace ngraph
