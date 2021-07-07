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
            class NGRAPH_API WrapType : public Pattern
            {
            public:
                static constexpr NodeTypeInfo type_info{"patternAnyType", 0};
                const NodeTypeInfo& get_type_info() const override;

                explicit WrapType(
                    NodeTypeInfo wrapped_type,
                    const ValuePredicate& pred = [](const Output<Node>& output) { return true; },
                    const OutputVector& input_values = {})
                    : Pattern(input_values, pred)
                    , m_wrapped_types({wrapped_type})
                {
                    set_output_type(0, element::Type_t::dynamic, PartialShape::dynamic());
                }

                explicit WrapType(
                    std::vector<NodeTypeInfo> wrapped_types,
                    const ValuePredicate& pred = [](const Output<Node>& output) { return true; },
                    const OutputVector& input_values = {})
                    : Pattern(input_values, pred)
                    , m_wrapped_types(std::move(wrapped_types))
                {
                    set_output_type(0, element::Type_t::dynamic, PartialShape::dynamic());
                }

                bool match_value(pattern::Matcher* matcher,
                                 const Output<Node>& pattern_value,
                                 const Output<Node>& graph_value) override;

                NodeTypeInfo get_wrapped_type() const;

                const std::vector<NodeTypeInfo>& get_wrapped_types() const;

            private:
                std::vector<NodeTypeInfo> m_wrapped_types;
            };
        } // namespace op

        template <class... Args>
        std::shared_ptr<Node> wrap_type(const OutputVector& inputs,
                                        const pattern::op::ValuePredicate& pred)
        {
            std::vector<DiscreteTypeInfo> info{Args::type_info...};
            return std::make_shared<op::WrapType>(info, pred, inputs);
        }

        template <class... Args>
        std::shared_ptr<Node> wrap_type(const OutputVector& inputs = {})
        {
            return wrap_type<Args...>(inputs, [](const Output<Node>& output) { return true; });
        }

        template <class... Args>
        std::shared_ptr<Node> wrap_type(const pattern::op::ValuePredicate& pred)
        {
            return wrap_type<Args...>({}, pred);
        }
    } // namespace pattern
} // namespace ngraph
