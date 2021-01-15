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
            class NGRAPH_API WrapType : public Pattern
            {
            public:
                static constexpr NodeTypeInfo type_info{"patternAnyType", 0};
                const NodeTypeInfo& get_type_info() const override;

                explicit WrapType(NodeTypeInfo wrapped_type,
                                  const ValuePredicate& pred =
                                      [](const Output<Node>& output) { return true; },
                                  const OutputVector& input_values = {})
                    : Pattern(input_values, pred)
                    , m_wrapped_type(wrapped_type)
                {
                    set_output_type(0, element::Type_t::dynamic, PartialShape::dynamic());
                }

                bool match_value(pattern::Matcher* matcher,
                                 const Output<Node>& pattern_value,
                                 const Output<Node>& graph_value) override;

                NodeTypeInfo get_wrapped_type() const { return m_wrapped_type; }
            private:
                NodeTypeInfo m_wrapped_type;
            };
        }

        template <class T>
        std::shared_ptr<Node> wrap_type(const OutputVector& inputs,
                                        const pattern::op::ValuePredicate& pred)
        {
            static_assert(std::is_base_of<Node, T>::value, "Unexpected template type");
            return std::make_shared<op::WrapType>(T::type_info, pred, inputs);
        }

        template <class T>
        std::shared_ptr<Node> wrap_type(const OutputVector& inputs = {})
        {
            return wrap_type<T>(inputs, [](const Output<Node>& output) { return true; });
        }

        template <class T>
        std::shared_ptr<Node> wrap_type(const pattern::op::ValuePredicate& pred)
        {
            return wrap_type<T>({}, pred);
        }
    }
}
