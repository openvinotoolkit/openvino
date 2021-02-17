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

#include <algorithm>
#include <regex>

#include "ngraph/pattern/op/pattern.hpp"

namespace ngraph
{
    namespace pattern
    {
        namespace op
        {
            // The symbols are required to be in cpp file to workaround RTTI issue on Android LLVM
            ValuePredicate Pattern::get_predicate() const { return m_predicate; }
            ValuePredicate as_value_predicate(NodePredicate pred)
            {
                if (pred == nullptr)
                {
                    return [](const Output<Node>&) { return true; };
                }
                else
                {
                    return [pred](const Output<Node>& value) {
                        return pred(value.get_node_shared_ptr());
                    };
                }
            }
        }

        PatternMap as_pattern_map(const PatternValueMap& pattern_value_map)
        {
            PatternMap result;
            for (auto& kv : pattern_value_map)
            {
                result[kv.first] = kv.second.get_node_shared_ptr();
            }
            return result;
        }

        PatternValueMap as_pattern_value_map(const PatternMap& pattern_map)
        {
            PatternValueMap result;
            for (auto& kv : pattern_map)
            {
                result[kv.first] = kv.second;
            }
            return result;
        }

        std::function<bool(Output<Node>)> consumers_count(size_t n)
        {
            return
                [=](Output<Node> output) -> bool { return output.get_target_inputs().size() == n; };
        }

        std::function<bool(Output<Node>)> has_static_dim(size_t pos)
        {
            return [=](Output<Node> output) -> bool {
                const auto& shape = output.get_partial_shape();
                return shape.rank().is_static() && shape.rank().get_length() > pos &&
                       shape[pos].is_static();
            };
        }

        std::function<bool(Output<Node>)> has_static_dims(const std::vector<size_t>& dims)
        {
            return [=](Output<Node> output) -> bool {
                const auto& shape = output.get_partial_shape();
                return shape.rank().is_static() &&
                       shape.rank().get_length() > *std::max_element(dims.begin(), dims.end()) &&
                       std::all_of(dims.begin(), dims.end(), [&shape](size_t pos) {
                           return shape[pos].is_static();
                       });
            };
        }

        std::function<bool(Output<Node>)> has_static_shape()
        {
            return
                [=](Output<Node> output) -> bool { return output.get_partial_shape().is_static(); };
        }

        std::function<bool(Output<Node>)> has_static_rank()
        {
            return [=](Output<Node> output) -> bool {
                return output.get_partial_shape().rank().is_static();
            };
        }

        std::function<bool(Output<Node>)> type_matches(const element::Type& type)
        {
            return [=](Output<Node> output) -> bool { return output.get_element_type() == type; };
        }

        std::function<bool(Output<Node>)>
            type_matches_any(const std::vector<element::Type>& expected_types)
        {
            return [=](Output<Node> output) -> bool {
                const auto& output_type = output.get_element_type();
                return std::any_of(expected_types.begin(),
                                   expected_types.end(),
                                   [=](element::Type type) { return type == output_type; });
            };
        }
    }
}
