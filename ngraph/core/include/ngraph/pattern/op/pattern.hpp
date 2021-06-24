// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <functional>

#include "ngraph/node.hpp"

namespace ngraph
{
    namespace pattern
    {
        namespace op
        {
            class Label;
        }

        class Matcher;
        class MatchState;

        using RPatternValueMap = std::map<std::shared_ptr<Node>, OutputVector>;
        using PatternValueMap = std::map<std::shared_ptr<Node>, Output<Node>>;
        using PatternValueMaps = std::vector<PatternValueMap>;

        using PatternMap = std::map<std::shared_ptr<Node>, std::shared_ptr<Node>>;

        PatternMap as_pattern_map(const PatternValueMap& pattern_value_map);
        PatternValueMap as_pattern_value_map(const PatternMap& pattern_map);

        template <typename T>
        std::function<bool(std::shared_ptr<Node>)> has_class()
        {
            auto pred = [](std::shared_ptr<Node> node) -> bool { return is_type<T>(node); };

            return pred;
        }

        NGRAPH_API
        std::function<bool(Output<Node>)> consumers_count(size_t n);

        NGRAPH_API
        std::function<bool(Output<Node>)> has_static_dim(size_t pos);

        NGRAPH_API
        std::function<bool(Output<Node>)> has_static_dims(const std::vector<size_t>& dims);

        NGRAPH_API
        std::function<bool(Output<Node>)> has_static_shape();

        NGRAPH_API
        std::function<bool(Output<Node>)> has_static_rank();

        NGRAPH_API
        std::function<bool(Output<Node>)> rank_equals(const Dimension& expected_rank);

        NGRAPH_API
        std::function<bool(Output<Node>)> type_matches(const element::Type& type);

        NGRAPH_API
        std::function<bool(Output<Node>)> type_matches_any(const std::vector<element::Type>& types);

        namespace op
        {
            using NodePredicate = std::function<bool(std::shared_ptr<Node>)>;
            using ValuePredicate = std::function<bool(const Output<Node>& value)>;

            NGRAPH_API
            ValuePredicate as_value_predicate(NodePredicate pred);

            class NGRAPH_API Pattern : public Node
            {
            public:
                /// \brief \p a base class for \sa Skip and \sa Label
                ///
                Pattern(const OutputVector& patterns, ValuePredicate pred)
                    : Node(patterns)
                    , m_predicate(pred)
                {
                    if (!m_predicate)
                    {
                        m_predicate = [](const Output<Node>&) { return true; };
                    }
                }

                Pattern(const OutputVector& patterns)
                    : Pattern(patterns, nullptr)
                {
                }

                virtual std::shared_ptr<Node>
                    clone_with_new_inputs(const OutputVector& /* new_args */) const override
                {
                    throw ngraph_error("Uncopyable");
                }

                ValuePredicate get_predicate() const;

            protected:
                ValuePredicate m_predicate;
            };
        } // namespace op
    }     // namespace pattern
} // namespace ngraph
