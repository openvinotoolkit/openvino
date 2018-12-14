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
#pragma once
#include "api/CPP/program.hpp"

#include "refcounted_obj.h"
#include "topology_impl.h"
#include "engine_impl.h"
#include "program_node.h"
#include "memory_impl.h"

#include <list>
#include <algorithm>

namespace cldnn
{

struct primitive_impl;
class layout_optimizer;
class constants_propagator;

/*
    cldnn_program implementation
*/
struct program_impl : public refcounted_obj<program_impl>
{
    friend struct program_node;

public:
    program_impl(engine_impl& engine_ref, topology_impl const& topology, build_options const& options, bool is_internal);

    void dump_memory_pool() const;

    engine_impl& get_engine() const { return *engine; }
    build_options get_options() const { return options; }
    bool is_debug_build() const { return options.get<build_option_type::debug>()->enabled(); }

    std::list<std::shared_ptr<program_node>> get_nodes() const;
    std::list<program_node*> get_processing_order() const { return processing_order; }
    std::list<primitive_id> get_optimized_out() const { return optimized_out; }
    program_node& get_node(primitive_id const& id)
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

    bool has_node(const primitive_id& prim) const
    {
        return nodes_map.count(prim) > 0;
    }

    program_node const& get_node(primitive_id const& id) const
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

private:
    uint32_t prog_id = 0;

    engine_impl::ptr engine;
    build_options options;

    std::list<program_node*> inputs;
    std::vector<program_node*> outputs;
    std::list<program_node*> processing_order;

    std::map<primitive_id, std::shared_ptr<program_node>> nodes_map;

    std::list<primitive_id> optimized_out;

    // TODO: Remove once we will get full support for input/output padding in all primitive implementations.
    bool output_size_handling_enabled;

    /*
    ** High-level functions, in order of usage
    */
    void init_graph(topology_impl const& topology);
    void pre_optimize_graph();
    void post_optimize_graph();
    void compile_graph();
    void cleanup();

    /*
    ** Initialization functions
    */
    void set_outputs();
    void calc_processing_order();
    void calc_prior_boxes();

    /*
    ** Analysis functions
    */
    void mark_constants();
    void mark_data_flow();
    // TODO: Remove once we will get full support for input/output padding in all primitive implementations.
    void analyze_output_size_handling_need();
    void replace_nodes_pre();
    void replace_nodes_post();
	void handle_lstm();
    void handle_reshape();

    /*
    ** Optimization functions
    */
    void trim_to_outputs();
    void remove_redundant_reorders();
    void calculate_BFS_processing_order();
    void reorder_inputs(layout_optimizer& lo);
    void pre_optimize_bias(layout_optimizer& lo);
    void post_optimize_weights(layout_optimizer& lo);
    void apply_needed_padding(program_node& node, program_node& prev_node, const padding& needed_padding);
    void prepare_padding();
    void propagate_constants();
    void prepare_buffer_fusing();
    void fuse_skip_layers(program_node* node);
    void prepare_primitive_fusing();
    void prepare_depthwise_sep_opt();
    void prep_opt_depthwise_sep_post();
    void update_processing_numbers();

    /*
    ** Memory pool functions
    */
    void prepare_memory_dependencies();
    void basic_memory_dependencies();
    void skipped_branch_memory_dependencies();
    void oooq_memory_dependencies();
    std::string get_memory_dependencies_string() const;

    /*
    ** Utilities
    */

    //returns already existing program_node for given primitive 'prim' (lookup in 'nodes_map')
    //if it was previously created, otherwise creates and then returns program_node
    program_node& get_or_create(std::shared_ptr<primitive> prim);

    // Inserts given program_node 'node' as an intermediate node between 'next' and it's
    //  dependency at 'prev_idx' index.
    void add_intermediate(program_node& node, program_node& next, size_t prev_idx, bool connect_int_node_with_old_dep = true);

    // Gets or creates program_node for given primitive 'prim' and inserts it as an intermediate
    // node between 'next' and it's dependency at 'prev_idx' index.
    void add_intermediate(std::shared_ptr<primitive> prim, program_node& next, size_t prev_idx, bool connect_int_node_with_old_dep = true)
    {
        add_intermediate(get_or_create(prim), next, prev_idx, connect_int_node_with_old_dep);
    }

    void add_connection(program_node& prev, program_node& next)
    {
        prev.users.push_back(&next);
        next.dependencies.push_back(&prev);
    }

    void remove_connection(program_node& prev, program_node& next)
    {
        prev.users.remove(&next);
        next.dependencies.erase(std::remove(next.dependencies.begin(), next.dependencies.end(), &prev), next.dependencies.end());
    }

    void remove_all_connections(program_node& node) {
        // since the graph is not topological sorted, we need to remove the node from both dependencies and users
        for (auto &e : node.users) {
            e->dependencies.erase(std::remove(e->dependencies.begin(), e->dependencies.end(), &node), e->dependencies.end());
        }
        for(auto &e : node.dependencies) {
            e->users.remove(&node);
        }
        node.dependencies.clear();
		node.users.clear();
    }

    bool processing_order_is_correct(program_node* node)
    {
        for (auto& dep : node->get_dependencies())
        {
            if (node->processing_num < dep->processing_num)
            {
                return false;
            }
        }
        return true;
    }

    void rename(program_node & node, primitive_id const & new_id);
    void swap_names(program_node& node1, program_node& node2);
    void replace_all_usages(program_node& old_node, program_node& new_node);

    //old_node - node which will be replaced
    //new_node - node which will replace the old one
    //replace_whole_branch - if set to true, 'old_node' will be replaced with all its dependencies and new_node will retain its dependencies
    //  old's dependencies which are post-dominates by 'old_node' will also be removed
    void replace(program_node& old_node, program_node& new_node, bool replace_whole_branch, bool check_output_layouts_integrity = true);

    //returns if 'node' has been removed
    bool remove_if_dangling(program_node& node, bool detach_whole_branch = false);

    //removes a node from the graph and deletes it afterwards,
    //prereq: node cannot be marked as output and has to have exactly one dependency
    //returns if 'node' has been extracted and removed successfully
    bool extract_and_remove(program_node& node);
    void replace_data_with_optimized(std::map<primitive_id, memory_impl::ptr> const& replace_map);
    void dump_program(const char* stage, bool with_full_info, std::function<bool(program_node const&)> const& filter = nullptr) const;
    //Dumps weights and biasses in serialization process, not working yet, in progress.
    void dump_weights_and_biasses(std::vector<unsigned long long>& offsets, std::vector<std::string>& data_names, std::ofstream& file_stream) const;
    //Makes serialization with given name.
    //Placeholder, not working yet, in progress.
    void serialize(std::string network_name, std::function<bool(program_node const&)> const& filter = nullptr) const;

    template <typename T>
    void optimize_bias(T& node, layout_optimizer& lo);

    template <typename T>
    void optimize_weights(T& node, layout_optimizer& lo);

    template <typename T>
    void optimize_depthwise_sep_pre(T& node);

    template <typename T>
    void optimize_depthwise_sep_post(T& node);
};
}

API_CAST(::cldnn_program, cldnn::program_impl)
