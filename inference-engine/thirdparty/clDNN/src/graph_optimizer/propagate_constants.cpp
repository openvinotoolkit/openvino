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
#include "program_node.h"
#include "engine_impl.h"
#include "program_impl.h"
#include "network_impl.h"
#include "data_inst.h"


using namespace cldnn;

//ToDo remove friendship relation from  program_node and program_impl
void propagate_constants::run(program_impl& p)
{
    for (auto& node : p.get_processing_order())
    {
        if (node->is_constant())
            handle_constant(p, *node);
    }

    auto&& to_replace = calculate(p.get_engine());

    //remove all nodes which are no longer relevant, i.e. nodes which:
    // 1. are constants, and
    // 2. do not have non-const user (so their data are not used during inference), and
    // 3. are not marked as outputs.
    // in case if node has either non-const user or is marked as output, it should be replace with cldnn::data rather than removed (see next loop)
    auto proc_itr = p.get_processing_order().begin();
    while (proc_itr != p.get_processing_order().end())
    {
        auto& node = (*proc_itr++);
        if (!node->is_constant())
            continue;
        if (has_non_const_user(*node) || (node->is_output() && !node->is_type<data>()))
            continue;

        auto& users = node->users;
        auto& deps = node->dependencies;

        for (size_t idx = 0; idx < deps.size(); idx++)
        {
            deps.at(idx)->users.remove(node);
        }
        deps.clear();

        for (auto& usr : users) {
            auto& usr_deps = usr->dependencies;
            usr_deps.erase(std::remove(usr_deps.begin(), usr_deps.end(), node), usr_deps.end());
        }
        users.clear();

        if (!node->is_output())
        {
            auto rem = p.remove_if_dangling(*node);
            assert(rem && "Non-output constant node which has only constant users should have been removed during constants propagation pass");
            (void)rem;
        }
    }

    //replace all constant nodes which are relevant for inference (either used by non-const user or marked as output) with recomputed cldnn::data
    for (auto& cout : to_replace)
    {
        auto& id_to_replace = cout.first;

        //TODO: do not use API primitives internally and get rid of this last 'cldnn::memory' internal usage
        memory api_memory = details::memory_c_to_cpp_converter::convert(api_cast(cout.second.get()));
        //c-cpp converter does not retain since normally it is done inside API-impl layer (cldnn.cpp) so we need to do it manually
        cout.second->add_ref();

        auto const_data = std::make_shared<data>("_cldnn_const_prop_" + id_to_replace, api_memory /* <<< REMOVE ME WHEN POSSIBLE */);
        auto& new_node = p.get_or_create(const_data);
        auto& curr_node = p.get_node(id_to_replace);

        if (!curr_node.is_type<generic_layer>())
        {
            auto curr_node_deps = curr_node.get_dependencies();
            for (auto& dep : curr_node_deps)
            {
                auto dep_users = dep->get_users();
                for (auto& dep_user : dep_users)
                {
                    if (dep_user == &curr_node)
                        p.remove_connection(*dep, curr_node);
                }
            }
        }

        curr_node.dependencies.clear();
        //remove all constant users (as they will be either removed or replaced by cldnn::data which does not have any dependencies)
        curr_node.users.erase(
            std::remove_if(curr_node.users.begin(), curr_node.users.end(), [](program_node* node) { return node->is_constant(); }),
            curr_node.users.end()
        );
        p.replace(curr_node, new_node);
    }
}

bool propagate_constants::has_non_const_user(program_node& node) const {
    if (!node.is_constant()) return true;
    for (auto &user : node.get_users())
    {
        if (!user->is_constant()) return true;
    }
    return false;
}

std::list<std::pair<primitive_id, memory_impl::ptr>> propagate_constants::calculate(engine_impl &engine)
{
    if (!has_non_trivial_constants)
        return{};

    build_options bo;
    bo.set_option(build_option::optimize_data(false));
    bo.set_option(build_option::outputs(const_outputs));
    network_impl::ptr net = engine.build_network(nodes, bo, true);
    for (auto& cin : const_inputs)
        net->set_input_data(cin->id(), cin->get_attached_memory());

    net->execute({});
    net->reset_execution(true); //wait for computations to complete
    auto outputs = net->get_outputs();

    std::list<std::pair<primitive_id, memory_impl::ptr>> ret;
    for (auto& out : outputs)
        ret.push_back({ out->id(), &out->output_memory() });

    return ret;
}

void propagate_constants::handle_constant(program_impl& prog, program_node& node)
{
    if (!node.is_type<data>())
    {
        add_constant(prog, node);
        if (has_non_const_user(node))
            const_outputs.push_back(node.id());
    }
}

void propagate_constants::add_constant(program_impl& prog, program_node& node)
{
    if (node.is_type<data>())
        return;
    nodes.insert(prog.get_node_ptr(node.get_primitive()->id));
    has_non_trivial_constants = true;

    //if a node is either an endpoint or an output, always add it as an output
    if (node.is_endpoint() || node.is_output())
        const_outputs.push_back(node.id());

    //if a non-tirivial constant has a trivial input, add this input as an input for our network
    add_deps_to_tpl(prog, node.get_dependencies());
}

void propagate_constants::add_deps_to_tpl(program_impl& prog, const std::vector<program_node*>& deps)
{
    /*
    Nodes can share dependencies, if we already have dep in tpl, don't add it again.
    example:
    C   <--- shared dep
    / \
    /   \
    A     B
    */
    for (auto& dep : deps)
    {
        if (dep->is_type<data>())
        {
            auto dep_ptr = prog.get_node_ptr(dep->get_primitive()->id);
            if (nodes.find(dep_ptr) == nodes.end())
            {
                nodes.insert(prog.get_node_ptr(dep->get_primitive()->id));
                const_inputs.push_back(&dep->as<data>());
            }
        }
    }
}