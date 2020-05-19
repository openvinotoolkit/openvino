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

#include <sstream>

#include "common_function_collection.hpp"

using namespace std;
using namespace ngraph;

pass::CommonFunctionCollection::CommonFunctionCollection(function<string(Node&, string)> emitter,
                                                         unordered_map<Node*, Node*>& result_map,
                                                         string& emitted_functions)
    : m_emit_op_as_function(emitter)
    , m_node_function_map(result_map)
    , m_emitted_functions(emitted_functions)
{
}

pass::CommonFunctionCollection::~CommonFunctionCollection()
{
}

bool pass::CommonFunctionCollection::run_on_module(vector<shared_ptr<Function>>& functions)
{
    // This for loop creates a collection of functions that are called more than once
    // and emitting them as globally callable functions.

    // match_function_map `key` contains the entire string of the function emitted for the
    // `value` Node*
    unordered_map<string, Node*> match_function_map;
    stringstream ss;
    const string function_name = "__f__";
    for (const shared_ptr<Function>& current_function : functions)
    {
        for (const shared_ptr<Node>& n : current_function->get_ordered_ops())
        {
            if (n->is_constant() || n->is_parameter())
            {
                continue;
            }
            if (n->is_op())
            {
                auto op = std::static_pointer_cast<op::Op>(n);
                auto annotations = op->get_op_annotations();
                // If an op is passed through, do not add it to the common function
                // collection so that the emitter can decide to eliminate it if desired
                if (annotations && annotations->get_in_place_oi_pairs().size() > 0)
                {
                    continue;
                }
            }

            Node& node = *n;

            // First emit the op as a function, something like this:
            // static void __f__(float* _arg0, float *_out1)
            // {
            //     op specific code here
            // }
            //
            // Then do a simple string compare in match_function_map to see if there is
            // another op that emits the exact same code.
            // If a match is found then the current node is mapped to call the original node's
            // function and the original node is *also* mapped to call the original node's function.
            // We also emit the static function declaration to m_emitted_functions when the match
            // is found the first time.
            string match_function = m_emit_op_as_function(node, function_name);
            auto it = match_function_map.find(match_function);
            if (it != match_function_map.end())
            {
                m_node_function_map.insert({&node, it->second});
                if (m_node_function_map.find(it->second) == m_node_function_map.end())
                {
                    m_node_function_map.insert({it->second, it->second});

                    // All of the functions are created with the same name `__f__` so here
                    // we rename it to something unique so we can compile everything when done.
                    auto offset = match_function.find(function_name);
                    string emitted_function = match_function;
                    string match_function_name = create_function_name(*it->second);
                    emitted_function.replace(offset, function_name.size(), match_function_name);
                    ss << emitted_function << "\n";
                }
            }
            else
            {
                match_function_map.insert({match_function, &node});
            }
        }
    }
    m_emitted_functions = ss.str();
    return false;
}

string pass::CommonFunctionCollection::create_function_name(const Node& node)
{
    return "func_" + node.get_name();
}
