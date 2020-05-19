//*****************************************************************************
// Copyright 2017-2020 Intel Corporation
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

#include "pattern.hpp"

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
    }
}
