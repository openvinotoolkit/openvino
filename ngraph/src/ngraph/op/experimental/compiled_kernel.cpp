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

#include "ngraph/op/experimental/compiled_kernel.hpp"

#include "ngraph/graph_util.hpp"
#include "ngraph/log.hpp"

using namespace std;
using namespace ngraph;

constexpr NodeTypeInfo op::CompiledKernel::type_info;

shared_ptr<Node>
    ngraph::op::CompiledKernel::clone_with_new_inputs(const OutputVector& new_args) const
{
    auto args = input_values();
    if (new_args.size() != args.size())
    {
        throw ngraph_error("number of arguments don't match");
    }

    // map inputs
    map<Output<Node>, Output<Node>> nm;
    for (size_t i = 0; i < args.size(); i++)
    {
        nm[args[i]] = new_args.at(i);
    }

    NodeVector new_node_list;
    for (auto n : m_node_list)
    {
        OutputVector cur_args;
        for (auto a : n->input_values())
        {
            if (as_type<op::Parameter>(a.get_node()))
            {
                // dummy parameter
                cur_args.push_back(a);
            }
            else
            {
                cur_args.push_back(nm.at(a));
            }
        }
        auto new_n = n->copy_with_new_inputs(cur_args);
        for (size_t i = 0; i < n->get_output_size(); ++i)
        {
            nm[n->output(i)] = new_n->output(i);
        }
        new_node_list.push_back(new_n);
    }

    OutputVector new_outputs;
    for (auto o : m_outputs)
    {
        new_outputs.push_back(nm.at(o));
    }

    auto ck = std::make_shared<CompiledKernel>(new_node_list, new_outputs, new_args);
    for (auto it : m_input_map)
    {
        ck->insert_to_input_map(it.first, it.second);
    }
    return std::move(ck);
}

ngraph::op::CompiledKernel::CompiledKernel(const NodeVector& node_list,
                                           const OutputVector& outputs,
                                           const OutputVector& args)
    : Op(args)
    , m_node_list(node_list)
    , m_outputs(outputs)
{
    constructor_validate_and_infer_types();
    encapsulate_nodes();
    set_output_size(m_outputs.size());

    for (size_t i = 0; i < outputs.size(); ++i)
    {
        auto& o = outputs.at(i);

        if (std::find(node_list.begin(), node_list.end(), o.get_node_shared_ptr()) ==
            node_list.end())
        {
            NODE_VALIDATION_CHECK(this, false, "Node for ", o, " isn't in node_list");
        }
        set_output_type(i, o.get_element_type(), o.get_partial_shape());
    }
}

void ngraph::op::CompiledKernel::encapsulate_nodes()
{
    std::unordered_set<std::shared_ptr<Node>> node_set(m_node_list.begin(), m_node_list.end());

    // Go through each non-CK user of input to CK
    int ck_arg_idx = 0;
    for (auto& arg_output : input_values())
    {
        for (auto& input : arg_output.get_target_inputs())
        {
            auto user = input.get_node();
            if (!as_type<op::CompiledKernel>(user) &&
                node_set.find(user->shared_from_this()) != node_set.end())
            {
                arg_output.remove_target_input(input);
                // Use a dummy Parameter as input for now, will replace later with the correct
                // one.
                auto temp_input_param = std::make_shared<ngraph::op::Parameter>(
                    arg_output.get_element_type(), arg_output.get_partial_shape());
                input.replace_source_output(temp_input_param->output(0));
                insert_to_input_map(temp_input_param, ck_arg_idx);
            }
        }
        ck_arg_idx++;
    }
}

void ngraph::op::CompiledKernel::insert_to_input_map(std::shared_ptr<Node> node, size_t ck_arg_idx)
{
    m_input_map.emplace(node, ck_arg_idx);
}
