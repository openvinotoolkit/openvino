// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/runtime/internal_properties.hpp"
#include "pass_manager.h"
#include "program_node.h"
#include "intel_gpu/runtime/engine.hpp"
#include "intel_gpu/runtime/debug_configuration.hpp"
#include "intel_gpu/graph/program.hpp"
#include "intel_gpu/graph/network.hpp"
#include "data_inst.h"
#include "intel_gpu/runtime/itt.hpp"
#ifdef ENABLE_ONEDNN_FOR_GPU
#include "reorder_inst.h"
#include "graph/impls/onednn/utils.hpp"
#endif // ENABLE_ONEDNN_FOR_GPU
#include <vector>
#include <list>
#include <memory>
#include <utility>

using namespace cldnn;

// ToDo remove friendship relation from  program_node and program
void propagate_constants::run(program& p) {
    OV_ITT_SCOPED_TASK(ov::intel_gpu::itt::domains::intel_gpu_plugin, "pass::PropagateConstants");
    for (auto& node : p.get_processing_order()) {
        if (node->is_constant())
            handle_constant(p, *node);
    }

    auto&& to_replace = calculate(p.get_engine(), p.get_config(), p.get_task_executor());

    // remove all nodes which are no longer relevant, i.e. nodes which:
    // 1. are constants, and
    // 2. do not have non-const user (so their data are not used during inference), and
    // 3. are not marked as outputs.
    // in case if node has either non-const user or is marked as output, it should be replace with cldnn::data rather
    // than removed (see next loop)
    auto proc_itr = p.get_processing_order().begin();
    while (proc_itr != p.get_processing_order().end()) {
        auto& node = (*proc_itr++);
        if (!node->is_constant())
            continue;
        if (has_non_const_user(*node) || (node->is_output() && !node->is_type<data>()))
            continue;

        auto& users = node->users;
        auto& deps = node->dependencies;

        for (size_t idx = 0; idx < deps.size(); idx++) {
            deps.at(idx).first->users.remove(node);
        }
        deps.clear();

        for (auto& usr : users) {
            auto& usr_deps = usr->dependencies;
            usr_deps.erase(std::remove_if(usr_deps.begin(), usr_deps.end(),
                           [&](const std::pair<program_node*, int>& dep) {
                               return node == dep.first;
                           }), usr_deps.end());
        }
        users.clear();

        if (!node->is_output()) {
            auto rem = p.remove_if_dangling(*node);
            assert(rem &&
                   "Non-output constant node which has only constant users should have been removed during constants "
                   "propagation pass");
            (void)rem;
        }
    }

    // replace all constant nodes which are relevant for inference (either used by non-const user or marked as output)
    // with recomputed cldnn::data
    for (auto& cout : to_replace) {
        auto& id_to_replace = std::get<0>(cout);
        auto mem_impl = std::get<1>(cout);
        auto cache_info = std::get<2>(cout);
        auto cache_manager = std::get<0>(cache_info);
        auto in_layout = std::get<1>(cache_info);
        auto reorder = std::get<2>(cache_info);

        auto const_data = std::make_shared<data>("_cldnn_const_prop_" + id_to_replace,
                                                 mem_impl, /* <<< REMOVE ME WHEN POSSIBLE */
                                                 cache_manager);
        auto& new_node = p.get_or_create(const_data);
        auto& curr_node = p.get_node(id_to_replace);

        // Remove dependencies
        auto curr_node_deps = curr_node.get_dependencies();
        for (auto& dep : curr_node_deps) {
            auto dep_users = dep.first->get_users();
            for (auto& dep_user : dep_users) {
                if (dep_user == &curr_node)
                    p.remove_connection(*dep.first, curr_node);
            }
        }

        if (in_layout && reorder) {
            new_node.as<data>().get_primitive()->cache_info->apply_reorder(in_layout, reorder);
        }

        curr_node.dependencies.clear();
        // remove all constant users (as they will be either removed or replaced by cldnn::data which does not have any
        // dependencies)
        curr_node.users.erase(std::remove_if(curr_node.users.begin(),
                                             curr_node.users.end(),
                                             [](program_node* node) { return node->is_constant(); }),
                              curr_node.users.end());
        p.replace(curr_node, new_node);
        new_node.recalc_output_layout(false);
    }
}

bool propagate_constants::has_non_const_user(program_node& node) const {
    if (!node.is_constant())
        return true;
    for (auto& user : node.get_users()) {
        if (!user->is_constant())
            return true;
    }
    return false;
}

using cache_tuple =
    std::tuple<std::shared_ptr<weightless_cache_manager>, std::shared_ptr<layout>, std::shared_ptr<reorder>>;

std::list<std::tuple<primitive_id, memory::ptr, cache_tuple>>
propagate_constants::calculate(engine& engine,
                               const ExecutionConfig& config,
                               std::shared_ptr<ov::threading::IStreamsExecutor> task_executor) {
    if (!has_non_trivial_constants)
        return {};

    ExecutionConfig cf_config = config.clone();
    cf_config.set_property(ov::intel_gpu::optimize_data(false));
    cf_config.set_property(ov::intel_gpu::custom_outputs(const_outputs));
    cf_config.finalize(engine);
    network::ptr net = network::build_network(engine, nodes, cf_config, task_executor, true);
    std::map<primitive_id, cache_tuple> weightless_cache_map;
    for (auto& cin : const_inputs) {
        net->set_input_data(cin->id(), cin->get_attached_memory_ptr());

        auto users = cin->get_users();
        if (users.size() == 1 && users.front()->is_type<reorder>()) {
            auto rprim = users.front()->as<reorder>().get_primitive();
            auto copy = std::shared_ptr<reorder>(new reorder(*rprim));
            auto id = rprim->id;
            auto cache_ptr = cin->as<data>().get_primitive()->cache_info;
            auto layout_ptr = std::make_shared<layout>(cin->get_output_layout());
            weightless_cache_map.emplace(id, std::make_tuple(cache_ptr, layout_ptr, copy));
        }
    }

    net->execute({});
    net->reset_execution(true);  // wait for computations to complete
    auto outputs = net->get_outputs();

    std::list<std::tuple<primitive_id, memory::ptr, cache_tuple>>
        ret;
    for (auto& out : outputs) {
        cache_tuple cache_info{};
        auto it = weightless_cache_map.find(out->id());
        if (it != weightless_cache_map.end()) {
            cache_info = it->second;
        }
        ret.push_back({out->id(), out->output_memory_ptr(), cache_info});
    }

    return ret;
}

void propagate_constants::handle_constant(program& prog, program_node& node) {
    if (!node.is_type<data>()) {
        add_constant(prog, node);
        if (has_non_const_user(node))
            const_outputs.push_back(node.id());
    }
}

void propagate_constants::add_constant(program& prog, program_node& node) {
    if (node.is_type<data>())
        return;
    nodes.insert(prog.get_node_ptr(node.get_primitive()->id));
    has_non_trivial_constants = true;

    // if a node is either an endpoint or an output, always add it as an output
    if (node.is_endpoint() || node.is_output())
        const_outputs.push_back(node.id());

    // if a non-tirivial constant has a trivial input, add this input as an input for our network
    add_deps_to_tpl(prog, node.get_dependencies());

#ifdef ENABLE_ONEDNN_FOR_GPU
    // Add reorder to transpose when the impl type of reorder is onednn and the weights for deconvolution should be transposed.
    bool is_reorder_weights = node.is_type<reorder>() && node.as<reorder>().get_primitive()->weights_reorder_params;
    if (is_reorder_weights) {
        const auto& weights_params = node.as<reorder>().get_primitive()->weights_reorder_params;
        auto onednn_weights_params = std::dynamic_pointer_cast<onednn::WeightsReorderParamsOneDNN>(weights_params);
        if (onednn_weights_params != nullptr && onednn_weights_params->should_be_transposed()) {
            auto& prev = node.get_dependency(0);
            cldnn::primitive_id rotate_reorder_id = prev.id() + "_rotate_reorder";
            auto grouped = weights_params->get_grouped();
            auto layout = weights_params->get_input_layout().convert_to_weights_layout(grouped);
            auto rotate_weights_params = std::make_shared<WeightsReorderParams>(layout, layout, true, grouped);
            auto rotate_prim = std::make_shared<cldnn::reorder>(rotate_reorder_id, prev.id(), rotate_weights_params);
            auto& rotate_node = prog.get_or_create(rotate_prim);
            prog.add_intermediate(rotate_node, node, 0);
            prog.get_or_create(rotate_prim).recalc_output_layouts(false);
            nodes.insert(prog.get_node_ptr(rotate_node.id()));
            GPU_DEBUG_LOG << "Added " << rotate_reorder_id << " for transposing weights before "
                << node.id() << std::endl;
        }
    }
#endif // ENABLE_ONEDNN_FOR_GPU
}

void propagate_constants::add_deps_to_tpl(program& prog, const std::vector<std::pair<program_node*, int32_t>>& deps) {
    /*
    Nodes can share dependencies, if we already have dep in tpl, don't add it again.
    example:
    C   <--- shared dep
    / \
    /   \
    A     B
    */
    for (auto& dep : deps) {
        if (dep.first->is_type<data>()) {
            auto dep_ptr = prog.get_node_ptr(dep.first->get_primitive()->id);
            if (nodes.find(dep_ptr) == nodes.end()) {
                nodes.insert(dep_ptr);
                const_inputs.push_back(&dep.first->as<data>());
            }
        }
    }
}
