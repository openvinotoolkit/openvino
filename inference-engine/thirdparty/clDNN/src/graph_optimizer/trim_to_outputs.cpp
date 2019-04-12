/*
// Copyright (c) 2018 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
*/

///////////////////////////////////////////////////////////////////////////////////////////////////

#include "pass_manager.h"

//ToDo: remove those include with the appropriate code below once we will have support for multiple outputs of a primitive
#include "batch_norm_inst.h"
#include "max_unpooling_inst.h"
#include "pooling_inst.h"

using namespace cldnn;

//This pass optimizes out nodes which have no impact on outputs
void trim_to_outputs::run(program_impl& p)
{
    const size_t actual_nodes = p.get_processing_order().size();
    if (!actual_nodes) //degenerated case but can happen
        return;

    if (p.get_outputs().size() == actual_nodes)
        return;

    //do backward bfs starting from all outputs
    std::list<const std::vector<program_node*>*> stack = { &(p.get_outputs()) };

    std::vector<program_node*> special_nodes;
    for (auto& node : p.get_processing_order())
    {
        if (node->is_type<input_layout>() ||  //input layout may become disconnected during prior boxes calculations so it may have not been marked at this place but we don't want to remove it
            node->is_type<max_unpooling>() || // ToDo: remove this after support for multi-outputs in primitives will be implemented.
            node->is_type<batch_norm>() ||
            (node->is_type<pooling>() && node->as<pooling>().get_primitive()->mode == pooling_mode::max_with_argmax))
                special_nodes.push_back(node);
    }
    stack.push_back(&special_nodes);

    while (!stack.empty())
    {
        auto nodes_list = stack.front();
        stack.pop_front();

        for (auto& node : *nodes_list)
        {
            if (!node->is_marked())
            {
                node->mark();
                if (!node->get_dependencies().empty())
                    stack.push_back(&node->get_dependencies());
            }
        }
    }

    //all not-marked nodes should be removed
    std::list<program_node*> to_rem;
    for (auto& node : p.get_processing_order())
    {
        if (!node->is_marked())
            to_rem.push_back(node);
    }
    p.remove_nodes(to_rem);
}