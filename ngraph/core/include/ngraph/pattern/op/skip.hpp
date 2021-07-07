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

                Skip(const OutputVector& args, ValuePredicate pred)
                    : Pattern(args, pred)
                {
                    set_output_type(
                        0, args.at(0).get_element_type(), args.at(0).get_partial_shape());
                }

                Skip(const OutputVector& args, NodePredicate pred = nullptr)
                    : Pattern(args, as_value_predicate(pred))
                {
                    set_output_type(
                        0, args.at(0).get_element_type(), args.at(0).get_partial_shape());
                }

                virtual bool match_value(pattern::Matcher* matcher,
                                         const Output<Node>& pattern_value,
                                         const Output<Node>& graph_value) override;
            };
        } // namespace op
    }     // namespace pattern
} // namespace ngraph
