/*
// Copyright (c) 2016 Intel Corporation
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

#include "error_handler.h"
#include "kernel_selector_helper.h"
#include "internal_primitive.h"
#include "internal_primitive_type_base.h"
#include "layout_optimizer.h"
#include "pass_manager.h"
#include "primitive_type.h"
#include "program_dump_graph.h"
#include "program_helpers.h"
#include "program_impl.h"
#include "sliding_window_utils.h"

#include "convolution_inst.h"
#include "concatenation_inst.h"
#include "crop_inst.h"
#include "data_inst.h"
#include "deconvolution_inst.h"
#include "detection_output_inst.h"
#include "input_layout_inst.h"
#include "lstm_inst.h"
#include "lstm_elt_inst.h"
#include "lstm_gemm_inst.h"
#include "mutable_data_inst.h"
#include "pooling_inst.h"
#include "primitive_inst.h"
#include "prior_box_inst.h"
#include "proposal_inst.h"
#include "reorder_inst.h"
#include "reshape_inst.h"
#include "split_inst.h"

#include "gpu/ocl_toolkit.h"

#include <fstream>
#include <algorithm>
#include <stdio.h>
#include <iostream>
#include <sstream>
#include <iomanip>
#include <memory>

program_impl::program_impl(engine_impl& engine_ref, topology_impl const& topology, build_options const& options, bool is_internal, bool no_optimizations)
    : engine(&engine_ref), options(options), processing_order(* new nodes_ordering), pm(std::unique_ptr<pass_manager>(new pass_manager()))
{
    set_options();
    prepare_nodes(topology);
    if (no_optimizations)
        init_graph();
    else
        build_program(is_internal);
}

program_impl::program_impl(engine_impl& engine_ref, std::set<std::shared_ptr<program_node>> const& nodes, build_options const& options, bool is_internal)
    : engine(&engine_ref), options(options), processing_order(*new nodes_ordering), pm(std::unique_ptr<pass_manager>(new pass_manager()))
{
    set_options();
    prepare_nodes(nodes);
    build_program(is_internal);
}

program_impl::~program_impl() = default;

program_node& program_impl::get_node(primitive_id const& id)
{
    try
    {
        return *nodes_map.at(id);
    }
    catch (...)
    {
        throw std::runtime_error("Program doesn't contain primtive node: " + id);
    }
}

program_node const& program_impl::get_node(primitive_id const& id) const
{
    try
    {
        return *nodes_map.at(id);
    }
    catch (...)
    {
        throw std::runtime_error("Program doesn't contain primtive node: " + id);
    }
}

// TODO: Remove once we will get full support for input/output padding in all primitive implementations.
bool program_impl::analyze_output_size_handling_need()
{
    bool handling_needed = false;

    // Calculate output size and compare with specified.
    for (const auto& node : processing_order)
    {
        if (node->is_type<convolution>())
        {
            auto& prim_node = node->as<convolution>();
            const auto& prim = prim_node.get_primitive();

            if (!prim->with_output_size)
                continue;

            tensor specified_output_range({ 0, 0, prim->output_size.spatial[0], prim->output_size.spatial[1] }, 1);

            auto filter_size = prim_node.weights(0).get_output_layout().size;

            auto calc_output_range = calc_sliding_window_output_range<swor_mode::all>(
                prim_node.input().get_output_layout().size,
                filter_size, prim->input_offset, prim->stride, prim->dilation, true, 1);

            if (specified_output_range != calc_output_range)
                handling_needed = true;
        }
        else if (node->is_type<deconvolution>())
        {
            auto& prim_node = node->as<deconvolution>();
            const auto& prim = prim_node.get_primitive();

            if (!prim->with_output_size)
                continue;

            tensor specified_output_range({ 0, 0, prim->output_size.spatial[0], prim->output_size.spatial[1] }, 1);

            auto filter_size = prim_node.weights(0).get_output_layout().size;

            auto calc_output_range = calc_sliding_window_needed_input_range(
                prim_node.input().get_output_layout().size,
                filter_size, prim->input_offset, prim->stride, { 1, 1, 1, 1 }, true, 1);

            if (specified_output_range != calc_output_range)
                handling_needed = true;
        }
        else if (node->is_type<pooling>())
        {
            auto& prim_node = node->as<pooling>();
            const auto& prim = prim_node.get_primitive();

            if (!prim->with_output_size)
                continue;

            tensor specified_output_range({ 0, 0, prim->output_size.spatial[0], prim->output_size.spatial[1] }, 1);

            // TODO: Check compatibility of output size calculation (with caffe).
            auto calc_output_range = calc_sliding_window_output_range<swor_mode::exceed_once_data>(
                prim_node.input().get_output_layout().size,
                prim->size, prim->input_offset, prim->stride, { 1, 1, 1, 1 }, true, 1);

            if (specified_output_range != calc_output_range)
                handling_needed = true;
        }
    }

    return handling_needed;
}

// create new nodes for a program based on the set of nodes
// method created to be used by propagate_constants to build sub program from constant nodes 
void program_impl::prepare_nodes(std::set<std::shared_ptr<program_node>>const &nodes)
{
    for (const auto& itr : nodes)
    {
        if (itr.get()->is_type<data>())
        {
            get_or_create(
                std::make_shared<input_layout>(itr.get()->id(), itr.get()->as<data>().get_primitive()->mem.get_layout())
            );
        }
        else
        {
            get_or_create(itr->desc);
        }
    }
    for (const auto& node : nodes_map)
    {
        auto node_ptr = node.second;
        if (node_ptr == nullptr)
            throw error("NULL pointer in nodes_map.", CLDNN_ERROR);
        //ToDo: avoid O(n^2) run time here (pass map instead of set?)
        bool found = false;
        for (const auto& src_node : nodes)
        {
            if (src_node == nullptr)
                throw error("NULL pointer in nodes_map.", CLDNN_ERROR);
            if (node.first == src_node->get_primitive()->id)
            {
                copy_node_dependencies(node_ptr.get(), src_node.get());
                found = true;
                break;
            }
        }
        if (!found)
        {
            add_node_dependencies(node_ptr.get());
        }
        if (node_ptr->dependencies.size() == 0)
            inputs.push_back(node_ptr.get());
    }
}

// create all nodes from topology primitives, add dependencies among them and create inputs list
void program_impl::prepare_nodes(topology_impl const &topology)
{
    auto const& topo_map = topology.get_primitives();
    for (const auto& prim : topo_map)
    {
        get_or_create(prim.second);
    }
    add_split_outputs();
    for (const auto& node : nodes_map)
    {
        auto node_ptr = node.second.get();
        if (node_ptr == nullptr)
            throw error("NULL pointer in nodes_map.", CLDNN_ERROR);
        add_node_dependencies(node_ptr);
        if (node_ptr->dependencies.size()==0)
        {
            inputs.push_back(node_ptr);
        }
    }
}

// add node's dependecies from its primitive dependencies
void program_impl::add_node_dependencies(program_node* node)
{
    auto deps = node->get_primitive()->dependencies();
    //add pointers to node's dependencies
    for (auto& dep : deps)
    {
        try {
            auto dep_node = nodes_map.at(dep);
            node->dependencies.push_back(dep_node.get());
            dep_node->users.push_back(node);
        }
        catch (...) {
            throw std::runtime_error("Program doesn't contain primitive: " + dep +
                " that is input to: " + node->get_primitive()->id);
        }
    }
}

/* helper method for program_impl constructor from list of nodes which
   copies src_node dependecies to the destination node dest_node dependencies.
   But only to those which appaer in this program implementation nodes_map */
void program_impl::copy_node_dependencies(program_node* dest_node, program_node* src_node)
{
    if (dest_node->get_primitive()->id != src_node->get_primitive()->id)
    {
        throw std::runtime_error("Node " + src_node->get_primitive()->id +  " and its copy " + dest_node->get_primitive()->id + " do not match.");
    }
    auto src_deps = src_node->get_dependencies();
    //add pointers to node's dependencies
    for (auto& src_dep : src_deps)
    {
        // do not copy dependencies to nodes which does not belong to the new (subgraph) topology
        if (nodes_map.find(src_dep->get_primitive()->id) == nodes_map.end()) continue;

        try {
            auto dest_dep = nodes_map.at(src_dep->get_primitive()->id);
            dest_node->dependencies.push_back(dest_dep.get());
            dest_dep->users.push_back(dest_node);
        }
        catch (...) {
            throw std::runtime_error("Program doesn't contain primitive: " + src_dep->get_primitive()->id +
                " that is input to: " + src_node->get_primitive()->id);
        }
    }
}

void program_impl::set_options()
{
    static std::atomic<uint32_t> id_gen{ 0 };
    prog_id = ++id_gen;
    assert(prog_id != 0);

    if ((options.get<build_option_type::tuning_config>()->config.mode == tuning_mode::tuning_tune_and_cache) &&
        !engine->configuration().enable_profiling)
    {
        throw std::invalid_argument("Engine must be created with profiling enabled in tune_and_cache mode!");
    }
}

void program_impl::build_program(bool is_internal)
{
    init_graph();
    {
        pre_optimize_graph(is_internal);
    }
    run_graph_compilation();
    {
        post_optimize_graph(is_internal);
    }
    engine->compile_program(*this);
    this->dump_program("finished", true);
    cleanup();
}

void program_impl::init_graph()
{
    graph_initializations graph_initializations_pass;
    pm->run(*this, graph_initializations_pass);

    calculate_prior_boxes calculate_prior_boxes_pass;
    pm->run(*this, calculate_prior_boxes_pass);
    
    mark_nodes mark_nodes_pass;
    pm->run(*this, mark_nodes_pass);
}

void program_impl::run_graph_compilation() {
    compile_graph compile_graph_pass;
    pm->run(*this, compile_graph_pass);
}

void program_impl::pre_optimize_graph(bool is_internal)
{
    trim_to_outputs trim_pass; //trim to outputs
    pm->run(*this, trim_pass); // ToDo remove hidden dependencies from trimm pass

    handle_input_padding handle_input_padding; // handle symmetric and asymmetric padding for input
    pm->run(*this, handle_input_padding);

    add_reshape_to_primitives add_reshape_to_primitives_pass; // add reshape to input/parameters for some primitives
    pm->run(*this, add_reshape_to_primitives_pass);

    processing_order.calculate_BFS_processing_order(); // this method makes sense only for OOOQ (out of order execution queue)

    bool output_size_handling_enabled = analyze_output_size_handling_need();
    for (auto& node : processing_order)
    {
        if (!node->is_type<internal_primitive>() && !node->is_type<data>())
            node->get_output_layout();
    }

    if (options.get<build_option_type::optimize_data>()->enabled())
    {
        prepare_primitive_fusing prepare_primitive_fusing_pass;
        pm->run(*this, prepare_primitive_fusing_pass);

        layout_optimizer lo(output_size_handling_enabled);
        reorder_inputs reorder_inputs_pass(lo);
        pm->run(*this, reorder_inputs_pass);

        // this code should be moved to post compilation after kernel selector will support handling reorder bias
        pre_optimize_bias pre_optimize_bias_pass(lo);
        pm->run(*this, pre_optimize_bias_pass);

        // passes regarding conv + eltwise optimizations

        // shrinking eltwise if users are conv 1x1 with stride > 1 optimization
        eltwise_shrinking eltwise_shrinking_pass;
        pm->run(*this, eltwise_shrinking_pass);

        // trying to set stride to 1x1 by shrinking convolutions before eltwise if doable
        eltwise_remove_stride eltwise_remove_stride_pass;
        pm->run(*this, eltwise_remove_stride_pass);

        prepare_conv_eltw_fusing prepare_conv_eltw_fusing_pass;
        pm->run(*this, prepare_conv_eltw_fusing_pass);

        prepare_conv_eltw_read_write_opt prepare_conv_eltw_read_write_opt_pass;
        pm->run(*this, prepare_conv_eltw_read_write_opt_pass);
    }

    handle_reshape();

    remove_redundant_reorders remove_redundant_reorders_pass;
    pm->run(*this, remove_redundant_reorders_pass);

    prepare_padding prepare_padding_pass(output_size_handling_enabled);
    pm->run(*this, prepare_padding_pass);

    prepare_depthwise_sep_opt prepare_depthwise_sep_opt_pass;
    pm->run(*this, prepare_depthwise_sep_opt_pass);

    if (!is_internal)
    {
        propagate_constants propagate_constants_pass;  // ToDo remove hidden dependencies from propagate_constants pass
        pm->run(*this, propagate_constants_pass);
    }

    //try to fuse buffers (i.e. depth_concat in bfyx format) after padding calculations
    if (options.get<build_option_type::optimize_data>()->enabled())
    {
        prepare_buffer_fusing prepare_buffer_fusing_pass;
        pm->run(*this, prepare_buffer_fusing_pass);
    }

    //check if there exists some layout incompatibilities and add an reorder node if required
    add_required_reorders add_required_reorders_pass;
    pm->run(*this, add_required_reorders_pass);
}

void program_impl::post_optimize_graph(bool is_internal)
{
    layout_optimizer lo;
    post_optimize_weights post_optimize_weights_pass(lo);
    pm->run(*this, post_optimize_weights_pass);

    remove_redundant_reorders remove_redundant_reorders_pass;
    pm->run(*this, remove_redundant_reorders_pass); //TODO: do we need it at this place also?

    if (!is_internal)
    {
        propagate_constants propagate_constants_pass;  // ToDo remove hidden dependencies from propagate_constants pass
        pm->run(*this, propagate_constants_pass);
    }

    prep_opt_depthwise_sep_post prep_opt_depthwise_sep_post_pass;
    pm->run(*this, prep_opt_depthwise_sep_post_pass);
   
    prepare_memory_dependencies();
}

// mark if the node is constant assuming that all dependencies are marked properly
void program_impl::mark_if_constant(program_node& node) 
{
    if (node.get_dependencies().empty())
        return;
    if (node.is_type<prior_box>())
        return;
    node.constant = true;
    for (auto& dep : node.get_dependencies())
    {
        if (!dep->constant)
        {
            node.constant = false;
            break;
        }
    }
}

// mark if the node is in data flow assuming that all dependencies are marked properly
void program_impl::mark_if_data_flow(program_node& node) 
{
    if (node.is_type<mutable_data>() || node.is_type<input_layout>())
    {
        node.data_flow = true;
    }
    else
    {
        node.data_flow = false;
        size_t inputs_count = node.get_dependencies().size();
        if (node.is_type<detection_output>() || node.is_type<proposal>())
            inputs_count = 2; //ignore third input as it is related to prior boxes (i.e. concat of prior-boxes)
        for (size_t idx = 0; idx < inputs_count; idx++)
        {
            if (node.get_dependency(idx).is_in_data_flow())
            {
                node.data_flow = true;
                return;
            }
        }
    }
}
void program_impl::cleanup()
{
    for (auto& node : processing_order)
        if (!node->is_type<internal_primitive>())
            node->get_output_layout();

    //in debug build, at the end, mark all nodes as outputs so user can query for buffers of all not-optimized nodes, including internal ones etc.
    if (is_debug_build())
    {
        for (auto& node : processing_order)
        {
            if (!node->is_output())
            {
                node->set_output(true);
                outputs.push_back(node);
            }
        }
    }
}

void program_impl::add_split_outputs() {
    auto itr = nodes_map.begin();
    while (itr != nodes_map.end())
    {
        auto node_itr = itr++;
        auto& node = (*node_itr).second;

        if (node->is_type<split>())
        {
            auto split_prim = node->as<split>().typed_desc();
            primitive_id input_id = split_prim->input[0];
            auto split_num = split_prim->output_offsets.size();

            //create crop for each split ouptut provided
            for (decltype(split_num) i = 0; i < split_num; i++)
            {
                primitive_id output_id = node->id() + ":" + split_prim->output_ids[i];

                //create dummy crop primitive and add it to nodes map
                auto crop_prim = std::make_shared<crop>(output_id, input_id, tensor{ 1,1,1,1 }, split_prim->output_offsets[i]);
                get_or_create(crop_prim);
            }
        }
    }
}

program_impl::nodes_ordering& program_impl::get_processing_order()
{
    return processing_order;
}

const program_impl::nodes_ordering& program_impl::get_processing_order() const
{
    return processing_order;
}

void add_memory_dependency(program_node* node, program_node* dep)
{
    if (node->can_be_optimized() ||
        !dep->can_be_optimized())
    {
        node->add_memory_dependency(dep->id());
    }
    else
    {
        if (node->id() == dep->id())
        {
            return;
        }
        for (auto subdep : dep->get_dependencies())
        {
            add_memory_dependency(node, subdep);
            add_memory_dependency(subdep, node);
        }
    }
}

void program_impl::basic_memory_dependencies()
{
    auto itr = processing_order.begin();
    std::vector<primitive_id> past_outputs;
    while (itr != processing_order.end())
    {
        auto& node = *itr;
        itr++;

        //data primitive can't be reused
        if (node->is_type<data>())
            continue;

        // add my dependencies to restriction list (can't share input.output buffers)
        for (auto it : node->get_dependencies())
        {
            add_memory_dependency(node, it);
            add_memory_dependency(it, node);
        }

        // Note we iterate over processing order, it means if primitve has processing num greater than any of outputs, this output
        // has to land on the primitve restriction list. Otherwise memory reuse can corrupt final results.
        node->add_memory_dependency(past_outputs);
        // if current node is an output add it to the outputs list after restriction.
        if (node->is_output())
            past_outputs.push_back(node->id());
    }
}


void program_impl::skipped_branch_memory_dependencies()
{
    // Primitive A can't use primitive B buffer if processing_num(B) < processing_num(A) and for any usr - the user of B processing_num(usr) > processing_num(A)
    // Otherwise it could override data that has to be used in the future.
    auto itrB = processing_order.begin();
    while (itrB != processing_order.end())
    {
        auto& nodeB = *itrB;
        auto itrA = ++itrB;
        if (nodeB->get_users().size()==0)
            continue;

        // find the last user of B in processing order
        auto itrUsr = nodeB->get_users().begin();
        auto lastUsr = itrUsr++;
        while (itrUsr != nodeB->get_users().end())
        {
            if (processing_order.get_processing_number(*lastUsr) < processing_order.get_processing_number(*itrUsr))
                lastUsr = itrUsr;
            itrUsr++;
        }

        //mark all nodes in between B and lastUsr of B as forbidden to share buffer with B
        while (itrA != processing_order.get_processing_iterator(**lastUsr))
        {
            auto& nodeA = *itrA;
            itrA++;
            add_memory_dependency(nodeA, nodeB);
            add_memory_dependency(nodeB, nodeA);
        }
    }
}

void program_impl::oooq_memory_dependencies()
{
    auto itr = processing_order.begin();
    // This order let us build dependencies based on syncing points.
    // Set of nodes between two syncing points will be called sync_region.
    // Major rules is: can't share resource with nodes in my sync_region

    int32_t last_barrier = 0;
    bool needs_barrier = false;
    std::vector<cldnn::program_node*> sync_region;
    while (itr != processing_order.end())
    {
        auto& node = *itr;
        itr++;

        // if any of dep has proccess num after barrier -> needs barrier
        for (auto dep : node->get_dependencies())
        {
            if (processing_order.get_processing_number(dep) >= last_barrier)
            {
                needs_barrier = true;
                break;
            }
        }

        if (needs_barrier)
        {
            last_barrier = processing_order.get_processing_number(node);
            needs_barrier = false;
            // add each pair bi-direction dependency
            for (auto nd1 = sync_region.begin(); nd1 + 1 != sync_region.end(); nd1++)
            {
                for (auto nd2 = nd1 + 1; nd2 != sync_region.end(); nd2++)
                {
                    add_memory_dependency(*nd1, *nd2);
                    add_memory_dependency(*nd2, *nd1);
                }
            }

            // collect dependencies of every node in sync region
            std::vector<cldnn::program_node*> deps;
            for (auto& nd_in_region : sync_region)
                for (auto& dep : nd_in_region->get_dependencies())
                    deps.emplace_back(dep);


            for (auto& nd_in_region : sync_region)
                for (auto& dep : deps)
                {
                    add_memory_dependency(nd_in_region, dep);
                    add_memory_dependency(dep, nd_in_region);
                }

            sync_region.clear();
        }
        sync_region.push_back(node);
    }
}

void program_impl::prepare_memory_dependencies()
{
    if (!get_engine().configuration().enable_memory_pool)
        return;

    basic_memory_dependencies();
    skipped_branch_memory_dependencies();
    oooq_memory_dependencies();
}

std::string program_impl::get_memory_dependencies_string() const
{
    std::string mem_dep = "Memory dependencies/restrictions:\n";
    auto itr = processing_order.begin();
    while (itr != processing_order.end())
    {
        auto& node = *itr;
        itr++;
        mem_dep = mem_dep.append("primitive: ").append(node->id()).append(" restricted list: ");
        for (auto it : node->get_memory_dependencies())
            mem_dep == mem_dep.append(it).append(", ");
        mem_dep = mem_dep.append("\n");
    }
    return mem_dep;
}

void program_impl::handle_reshape()
{
    //reshape primitive by definition does not change underlying data, only shape description
    //however during graph initialization and data optimization the layouts can be changed without user's knowledge,
    //when reshape is followed by reorder, it is likely that reorder's output will not be as expected (for example reshape with flattened shape)
    //this pass resolved the issue by changing graph in the following way
    //- in case reshape has multiple users with reshape->reorder sequence, it will be splitted to multiple reshape primitives with single user
    //- in case of reshape->reorder sequence, the additional reorder before reshape will be added,
    //  if last reorder does not contain padding or mean subtract, it will be removed later in the graph

    for (const auto& node : processing_order)
    {
        if (node->is_type<reshape>())
        {
            auto& input_node = node->get_dependency(0);

            if (input_node.is_type<reorder>())
                continue;

            node->get_output_layout();
            if (node->as<reshape>().is_in_place())
                node->optimized = true;

            //vector for storing nodes that are reorder type, for which splitted primitives are needed (except for the first one where orginal reshape will be used)
            std::vector<program_node*> reorder_node_to_split;

            //find the users of reshape that are reorder type, if none present then skip the current node
            for (const auto& user : node->get_users())
            {
                if (user->is_type<reorder>())
                    reorder_node_to_split.push_back(user);
            }

            if (!reorder_node_to_split.empty())
            {
                auto& prim_node = node->as<reshape>();
                const auto& prim = prim_node.get_primitive();
                auto output_shape = prim->output_shape;

                //vector for storing reshape nodes to connect to new reorder nodes (if needed)
                std::vector<program_node*> reorder_reshape_nodes;

                bool skip_first_user = false;
                auto reshape_users = node->get_users();
                for (const auto& user : reshape_users)
                {
                    //reshape node for first user will be the orginal reshape from the graph
                    if (!skip_first_user)
                    {
                        if (std::find(reorder_node_to_split.begin(), reorder_node_to_split.end(), user) != reorder_node_to_split.end())
                            reorder_reshape_nodes.push_back(node);
                        skip_first_user = true;
                        continue;
                    }

                    //other reshapes will be clones of the orginal one connected to reshape->reorder sequences
                    if (std::find(reorder_node_to_split.begin(), reorder_node_to_split.end(), user) != reorder_node_to_split.end())
                    {
                        auto new_reshape = std::make_shared<reshape>("_reshape_split_" + user->id() + "_" + node->id(), input_node.id(), output_shape);
                        auto& new_reshape_node = get_or_create(new_reshape);
                        add_connection(input_node, new_reshape_node);
                        user->replace_dependency(0, new_reshape_node);
                        processing_order.insert_next(&input_node, &new_reshape_node);
                        reorder_reshape_nodes.push_back(&new_reshape_node);
                    }
                }

                //add new reorder nodes to proper reshape node
                auto reshape_reorder_id = 0;
                for (const auto& reorder_node : reorder_node_to_split)
                {
                    /*
                    auto new_reshape = std::make_shared<reshape>("_reshape_split_" + user->id() + "_" + node->id(), input_node.id(), output_shape);
                    auto& new_reshape_node = get_or_create(new_reshape);
                    add_connection(input_node, new_reshape_node);
                    user->replace_dependency(0, new_reshape_node);
                    processing_order.insert(std::next(processing_order.get_processing_iterator(input_node)), &new_reshape_node);
                    reorder_reshape_nodes.push_back(&new_reshape_node);
                    */
                    auto& reorder_reshape_node = reorder_reshape_nodes[reshape_reorder_id];
                    auto reshape_in_layout = reorder_node->get_output_layout();
                    auto reshape_input = std::make_shared<reorder>("_reshape_input_" + reorder_node->id() + "_" + reorder_reshape_node->id(), input_node.id(),
                        reshape_in_layout.format, reshape_in_layout.data_type);
                    auto& reshape_input_node = get_or_create(reshape_input);
                    add_intermediate(reshape_input_node, *reorder_reshape_node, 0, reshape_input_node.dependencies.empty());
                    reshape_reorder_id++;
                }
            }

            auto reshape_layout = node->get_output_layout();
            if (!(node->is_output()) && (reshape_layout.format != cldnn::format::bfyx))
            {
                auto bfyx_layout = layout({ reshape_layout.data_type, cldnn::format::bfyx, reshape_layout.size });
                //when some primitive does an implicit reorder to some other format then we lose the info about pitches in reshape stage
                //we assume user provides the input vector in bfyx
                if (!program_helpers::are_layouts_identical(reshape_layout, bfyx_layout).second)
                {
                    auto reshape_input = std::make_shared<reorder>("_reshape_input_" + node->id(), input_node.id(), cldnn::format::bfyx, reshape_layout.data_type);
                    auto& reshape_input_node = get_or_create(reshape_input);
                    add_intermediate(reshape_input_node, *node, 0, reshape_input_node.dependencies.empty());

                    auto reshape_users = node->get_users();
                    for (const auto& user : reshape_users)
                    {
                        size_t idx = 0;
                        for (size_t i = 0; i < user->get_dependencies().size(); i++)
                        {
                            auto& input = user->get_dependency(i);
                            if (input.id() == node->id()) {
                                idx = i;
                                break;
                            }
                        }
                        auto reshape_output = std::make_shared<reorder>("_reshape_output_" + node->id(), user->id(), reshape_layout.format, reshape_layout.data_type);
                        auto& reshape_output_node = get_or_create(reshape_output);
                        add_intermediate(reshape_output_node, *user, idx, reshape_output_node.dependencies.empty());
                    }
                }
            }
        }
    }
}

void program_impl::apply_needed_padding(program_node& node, program_node& prev_node,
    const padding& needed_padding)
{
    auto target_layout = prev_node.get_output_layout();

    // Short circuit if padding did not change.
    if (target_layout.data_padding == needed_padding)
        return;

    // Special handling for input nodes.
    if (prev_node.is_type<input_layout>() || prev_node.is_type<mutable_data>())
    {
        target_layout.data_padding = needed_padding;

        auto r_prim = std::make_shared<reorder>("reorder_input_" + node.id(), prev_node.id(), target_layout);
        add_intermediate(r_prim, node, 0);
        return;
    }

    prev_node.merge_output_padding(needed_padding);
}

void program_impl::reverse_connection(program_node& dep_node, program_node& user_node)
{
    if (std::find(dep_node.users.begin(), dep_node.users.end(), &user_node) != dep_node.users.end())
    {
        remove_connection(dep_node, user_node);
        add_connection(user_node, dep_node);
    }
    else
        throw std::runtime_error("Trying to reverse connection, but nodes are wrongly or not connected.");
}

program_node& program_impl::get_or_create(std::shared_ptr<primitive> prim)
{
    auto itr = nodes_map.lower_bound(prim->id);
    if (itr != nodes_map.end() && itr->first == prim->id)
        return *itr->second;

    auto new_node = prim->type->create_node(*this, prim);
    nodes_map.insert(itr, { prim->id, new_node });
    return *new_node;
}

void program_impl::add_intermediate(program_node& node, program_node& next, size_t prev_idx,
    bool connect_int_node_with_old_dep, bool move_usrs_of_prev_to_node)
{
    if (connect_int_node_with_old_dep && !node.dependencies.empty())
        throw std::invalid_argument("Node which is about to be added in between two other nodes should not have any existing dependencies");

    auto& prev = next.get_dependency(prev_idx);
    //firstly add connection, later replace dependency, so 'prev' won't become dangling and therefore removed
    if (connect_int_node_with_old_dep)
    {
        add_connection(prev, node);
        if (processing_order.size() != 0)
        {
            processing_order.insert_next(&prev, &node);
        }
    }

    if (move_usrs_of_prev_to_node) {
        auto itr = prev.get_users().begin();
        while(itr!= prev.get_users().end())
        {
            auto usr = *itr;
            itr++;
            if (usr->id() != node.id())
                usr->replace_dependency(prev, node);
        }
        mark_if_constant(prev);
        mark_if_constant(node);
        mark_if_data_flow(prev);
        mark_if_data_flow(node);
    }
    else {
        next.replace_dependency(prev_idx, node);
        node.constant = prev.constant;
        node.data_flow = prev.data_flow;
    }
}

void program_impl::add_intermediate(std::shared_ptr<primitive> prim, program_node& next, size_t prev_idx, 
    bool connect_int_node_with_old_dep, bool move_usrs_of_prev_to_node)
{
    add_intermediate(get_or_create(prim), next, prev_idx, connect_int_node_with_old_dep, move_usrs_of_prev_to_node);
}

void program_impl::add_connection(program_node& prev, program_node& next)
{
    prev.users.push_back(&next);
    next.dependencies.push_back(&prev);
}

void program_impl::remove_connection(program_node& prev, program_node& next)
{
    prev.users.remove(&next);
    next.dependencies.erase(std::remove(next.dependencies.begin(), next.dependencies.end(), &prev), next.dependencies.end());
}

void program_impl::remove_all_connections(program_node& node) {
    // since the graph is not topological sorted, we need to remove the node from both dependencies and users
    for (auto &e : node.users)
    {
        e->dependencies.erase(std::remove(e->dependencies.begin(), e->dependencies.end(), &node), e->dependencies.end());
    }
    for (auto &e : node.dependencies) 
    {
        e->users.remove(&node);
    }
    node.dependencies.clear();
    node.users.clear();
}

void program_impl::rename(program_node & node, primitive_id const & new_id)
{
    if (nodes_map.count(new_id))
        throw std::runtime_error("Trying to rename program_node but node with id " + new_id + " already exists");
    if (node.is_output())
        throw std::invalid_argument("Trying to rename an output node. If you intend to do that, please clear 'output' flag manually.");

    auto node_ptr = nodes_map.find(node.id())->second;
    nodes_map.emplace(new_id, node_ptr);
    nodes_map.erase(node.id());

    if (!node.is_type<internal_primitive>())
        const_cast<primitive_id&>(node.desc->id) = new_id;
    else
        reinterpret_cast<details::internal_program_node_base&>(node).internal_id = new_id;
}

void program_impl::swap_names(program_node& node1, program_node& node2)
{
    const auto _extract_id = [](program_node& node) -> primitive_id&
    {
        if (!node.is_type<internal_primitive>())
            return const_cast<primitive_id&>(node.desc->id);
        else
            return reinterpret_cast<details::internal_program_node_base&>(node).internal_id;
    };

    nodes_map.at(node1.id()).swap(nodes_map.at(node2.id()));
    std::swap(_extract_id(node1), _extract_id(node2));
}

void program_impl::replace_all_usages(program_node & old_node, program_node & new_node)
{
    auto itr = old_node.users.begin();
    bool end = (itr == old_node.users.end());
    while (!end)
    {
        auto& usage = (*itr++);
        end = (itr == old_node.users.end());
        usage->replace_dependency(old_node, new_node);
    }
}

void program_impl::replace(program_node& old_node, program_node& new_node)
{
    if (!new_node.dependencies.empty() || !new_node.users.empty())
        throw std::invalid_argument("Node which is about to replace other node should be detached");

    if (new_node.is_output())
        throw std::invalid_argument("Replacement node shouldn't be marked as an output since it's impossible to rename such node.");

    auto id = old_node.id();
    new_node.output_layout = old_node.get_output_layout();
    new_node.valid_output_layout = old_node.valid_output_layout;

    
    //copy old's dependencies
    while (!old_node.dependencies.empty())
    {
        auto& dep = old_node.dependencies.front();
        add_connection(*dep, new_node);
        remove_connection(*dep, old_node);
    }

    //append users
    for (auto& user : old_node.users)
    {
        new_node.users.push_back(user);
        for (auto& users_dep : user->dependencies)
        {
            if (users_dep == &old_node)
            {
                users_dep = &new_node;
                break;
            }
        }
    }

    old_node.users.clear();

    bool old_was_output = false;
    //copy node's state
    if (old_node.is_output())
    {
        old_was_output = true;
        old_node.set_output(false);
        outputs.erase(std::remove(outputs.begin(), outputs.end(), &old_node), outputs.end());
    }
    if (new_node.is_input())
        inputs.push_back(&new_node);
    if (old_node.is_input())
        inputs.remove(&old_node);

    new_node.constant = old_node.constant;
    new_node.user_mark = old_node.user_mark;

    processing_order.insert(&old_node, &new_node);
    if (processing_order.get_processing_iterator(old_node) != processing_order.end())
        processing_order.erase(&old_node);
    nodes_map.erase(id);
    rename(new_node, id);

    //mark new node as an output after renaming
    if (old_was_output)
    {
        new_node.set_output(true);
        outputs.push_back(&new_node);
    }
}

bool program_impl::remove_if_dangling(program_node& node)
{
    if (!node.users.empty())
        return false;
    if (!node.dependencies.empty())
        return false;

    if (!node.is_output() || is_debug_build())
    {
        if (node.is_input())
            inputs.remove(&node);

        if (std::find(processing_order.begin(), processing_order.end(), &node) != processing_order.end())
            processing_order.erase(&node);
        optimized_out.push_back(node.id());
        nodes_map.erase(node.id());
    }
    return true;
}

bool program_impl::extract_and_remove(program_node& node)
{
    if (node.get_dependencies().size() != 1)
        return false;

    if (node.is_output() && node.get_dependency(0).is_output() && !is_debug_build()) //TODO: add a mechanism to support removal of nodes which are marked as outputs
        return false;

    if (node.is_output() && !is_debug_build())
    {
        auto& prev = node.get_dependency(0);
        auto node_id = node.id();

        node.set_output(false);
        outputs.erase(std::remove(outputs.begin(), outputs.end(), &node), outputs.end());

        rename(node, "_cldnn_tmp_" + node_id);
        rename(prev, node_id);

        prev.set_output(true);
        outputs.push_back(&prev);
    }

    auto& input = node.get_dependency(0);
    node.dependencies.clear();
    input.users.remove(&node);

    if (!node.is_endpoint())
        replace_all_usages(node, input);
    else
        remove_if_dangling(node);

    return true;
}

void program_impl::remove_nodes(std::list<program_node*>& to_remove)
{
    for (auto const& node : to_remove)
    {
        if (node->is_input())
            get_inputs().remove(node);
        else
        {
            for (auto& dep : node->dependencies)
                dep->users.remove(node);
        }
        for (auto& user : node->users)
        {
            user->dependencies.erase(std::remove(user->dependencies.begin(),
                user->dependencies.end(), node),
                user->dependencies.end());
        }
        get_processing_order().erase(node);
        optimized_out.push_back(node->id());
        nodes_map.erase(node->id());
    }
}

void program_impl::dump_memory_pool() const
{
    if (!get_engine().configuration().enable_memory_pool)
        return;
    auto path = get_dir_path(options);
    if (path.empty())
    {
        return;
    }
    path += "cldnn_memory_pool.log";
    auto dep = get_memory_dependencies_string();
    get_engine().dump_memory_pool(*this, path, dep);
    std::string dump_file_name = std::to_string(pm->get_pass_count()+1) + "_memory_pool";
    dump_program(dump_file_name.c_str(), true);
}

//TODO: break this function into number of smaller ones + add per-primitive fields (possibly use primitive_inst::to_string?)
void program_impl::dump_program(const char* stage, bool with_full_info, std::function<bool(program_node const&)> const& filter) const
{
    std::string path = get_dir_path(options);
    if (path.empty())
    {
        return;
    }

    std::ofstream graph(path + "cldnn_program_" + std::to_string(prog_id) + "_" + stage + ".graph");
    dump_graph_init(graph, *this, filter);

    if (!with_full_info)
    {
        return;
    }

    graph.open(path + "cldnn_program_" + std::to_string(prog_id) + "_" + stage + ".info");
    dump_graph_info(graph, *this, filter);

    graph.open(path + "cldnn_program_" + std::to_string(prog_id) + "_" + stage + ".order");
    dump_graph_processing_order(graph, *this);

    graph.open(path + "cldnn_program_" + std::to_string(prog_id) + "_" + stage + ".optimized");
    dump_graph_optimized(graph, *this);
}


