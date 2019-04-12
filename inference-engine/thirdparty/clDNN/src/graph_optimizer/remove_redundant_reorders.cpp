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
#include "program_helpers.h"

using namespace cldnn;

void remove_redundant_reorders::run(program_impl& p)
{
    auto itr = p.get_processing_order().begin(); //note we need to use iterators since currently processed element can be removed
    while (itr != p.get_processing_order().end())
    {
        auto& node = (*itr++); //post-inc to avoid invalidation due to possible erase
        if (!node->is_type<reorder>()) //only care for reorders
            continue;

        program_node* current_node = node;
        std::vector<program_node*> r_nodes_to_remove;

        auto optimize = true;
        while (current_node)
        {
            auto& r_node = current_node->as<reorder>();
            current_node = nullptr;

            if (r_node.has_mean() || !r_node.get_primitive()->subtract_per_feature.empty()  //do not optimize if mean of subtract are present
                || r_node.is_output()) //do not optimize when both reorder and layer before are outputs
            {
                optimize = false;
                break;
            }

            r_nodes_to_remove.push_back(&r_node);

            if (r_node.get_dependency(0).is_type<reorder>() && r_node.get_dependencies().size() == 1 && r_node.get_users().size() == 1 && r_node.get_dependency(0).get_users().size() == 1)
                current_node = &r_node.get_dependency(0);
        }
        if (!optimize)
            continue;

        assert(node->get_dependencies().size() == 1 && "reorder without mean should have exactly one dependecy (input)");
        auto& r_output = r_nodes_to_remove.front();
        auto& r_input = r_nodes_to_remove.back()->get_dependency(0);
        auto o_layout = r_output->get_output_layout();
        auto i_layout = r_input.get_output_layout();

        auto ident = program_helpers::are_layouts_identical(o_layout, i_layout);
        if (!ident.second)
            continue;

        for (auto remove_reorder_node : r_nodes_to_remove)
        {
            auto& r_node = remove_reorder_node->as<reorder>();

            if (ident.first && ident.second && r_node.is_output() && r_node.get_dependency(0).is_input()) //do not optimize when reorder is output and layer before is input
            {
                optimize = false;
                break;
            }
        }
        if (!optimize)
            continue;

        auto rem_itr = r_nodes_to_remove.begin();
        while (rem_itr != r_nodes_to_remove.end())
        {
            auto remove_reorder_node = *rem_itr++;
            auto& r_node = remove_reorder_node->as<reorder>();
            //mark as optimized
            r_node.can_be_optimized(true);
            r_node.requires_reinterpret(!ident.first);
            if (ident.first) //no need of reshape
                p.extract_and_remove(r_node); //try to remove if possible (with respect to r_node not being marked as output)
        }
    }
}
