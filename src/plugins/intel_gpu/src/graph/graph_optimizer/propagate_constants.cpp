// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <algorithm>

#include "data_inst.h"
#include "debug_helper.hpp"
#include "impls/ocl/kernels_cache.hpp"
#include "intel_gpu/graph/network.hpp"
#include "intel_gpu/graph/program.hpp"
#include "intel_gpu/graph/topology.hpp"
#include "intel_gpu/runtime/debug_configuration.hpp"
#include "intel_gpu/runtime/engine.hpp"
#include "intel_gpu/runtime/itt.hpp"
#include "mutable_data_inst.h"
#include "pass_manager.h"
#include "primitive_inst.h"
#include "program_node.h"
#include "registry/implementation_manager.hpp"
#include "reorder_inst.h"
#ifdef ENABLE_ONEDNN_FOR_GPU
#    include "graph/impls/onednn/utils.hpp"
#endif  // ENABLE_ONEDNN_FOR_GPU
#include <cassert>
#include <memory>
#include <queue>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

using namespace cldnn;

namespace {
// Attempts to reselect an appropriate implementation for a node after
// propagate_constants transforms dynamic inputs into static data.
// Refreshes stale output layouts before building kernel params to avoid
// incorrect shape_type classification.
void try_reselect_impl_for_node(program_node* node) {
    bool can_select_impl = !node->is_type<data>() && (!node->is_type<mutable_data>() || !node->get_dependencies().empty());
    if (!can_select_impl) {
        return;
    }

    auto* selected_impl = node->get_selected_impl();
    bool has_selected_impl = selected_impl != nullptr;
    bool need_new_impl_selection = !has_selected_impl;

    if (has_selected_impl) {
        bool is_node_dynamic = node->get_output_layout(false).is_dynamic();
        bool is_impl_dynamic = selected_impl->is_dynamic();
        need_new_impl_selection = (is_node_dynamic != is_impl_dynamic);
    }

    if (!need_new_impl_selection) {
        return;
    }

    // Refresh stale output layouts before building kernel params.
    // After invalidate_users(), cached output_layouts may still reflect
    // the old dynamic shape. Recomputing ensures get_kernel_impl_params()
    // uses up-to-date layouts for accurate shape_type determination.
    if (!node->is_all_valid_output_layouts()) {
        node->get_output_layouts(false);
    }

    auto params = node->get_kernel_impl_params();
    auto shape_type = ImplementationManager::get_shape_type(*params);
    if (shape_type == shape_types::dynamic_shape) {
        return;
    }

    auto selected_impl_manager = node->type()->choose_impl(*node, shape_type);
    std::string fail_reason;
    try {
        if (selected_impl_manager) {
            node->set_selected_impl(selected_impl_manager->create(*node, *params));
        } else {
            fail_reason = "choose_impl returned nullptr (no matching implementation found)";
        }
    } catch (const std::exception& e) {
        fail_reason = e.what();
    }

    OPENVINO_ASSERT(node->get_selected_impl() != nullptr,
                    "[GPU] Failed to select implementation after propagate_constants"
                    "\nname:",
                    node->id(),
                    "\ntype: ",
                    node->get_primitive()->type_string(),
                    "\noriginal_type: ",
                    node->get_primitive()->origin_op_type_name,
                    " ",
                    fail_reason);
}

bool is_computed_constant(const program_node& node) {
    return node.is_constant() && !node.is_type<data>();
}

}  // namespace

// ToDo remove friendship relation from  program_node and program
void propagate_constants::run(program& p) {
    OV_ITT_SCOPED_TASK(ov::intel_gpu::itt::domains::intel_gpu_plugin, "pass::PropagateConstants");
    auto subgraphs = find_constant_subgraphs(p);
    std::unordered_set<program_node*> reselection_targets;
    if (!subgraphs.empty()) {
        // build_kernels makes dynamic constant nodes static; their users are reselected once the constants are data
        std::unordered_set<program_node*> dynamic_nodes;
        for (const auto& subgraph : subgraphs) {
            for (auto* node : subgraph) {
                if (node->get_output_layout().is_dynamic()) {
                    dynamic_nodes.insert(node);
                }
            }
        }
        // An empty program provides the kernels cache, the stream and the memory allocation for the instances of the
        // constant nodes. The custom outputs name nodes of the main program, which this one does not have.
        ExecutionConfig context_config = p.get_config().clone();
        context_config.set_property(ov::intel_gpu::custom_outputs(std::vector<std::string>{}));
        auto context = program::build_program(p.get_engine(), topology(), context_config, p.get_task_executor(), true);
        build_kernels(subgraphs, context->get_kernels_cache());
        auto net = network::allocate_network(p.get_engine(), context, true, true);

        for (const auto& [begin, end] : make_batches(subgraphs)) {
            std::vector<program_node*> inputs;
            for (size_t s = begin; s < end; ++s) {
                for (auto* node : subgraphs[s]) {
                    for (const auto& dep : node->get_dependencies()) {
                        if (dep.first->is_type<data>() && std::find(inputs.begin(), inputs.end(), dep.first) == inputs.end()) {
                            inputs.push_back(dep.first);
                        }
                    }
                }
            }
            auto outputs = execute_subgraphs(*net, inputs, subgraphs, begin, end);

            std::unordered_set<program_node*> replaced;
            std::vector<cache_tuple> cache_info;
            for (const auto& output : outputs) {
                replaced.insert(output.first);
                cache_info.push_back(get_weightless_cache_info(*output.first));
            }
            for (size_t s = begin; s < end; ++s) {
                for (auto* node : subgraphs[s]) {
                    if (replaced.count(node) == 0) {
                        remove_node(p, *node);
                    }
                }
            }
            for (size_t i = 0; i < outputs.size(); ++i) {
                const bool was_dynamic = dynamic_nodes.count(outputs[i].first) != 0;
                replace_with_data(p, *outputs[i].first, outputs[i].second, cache_info[i], was_dynamic, reselection_targets);
            }
            // Weights read by no other node are freed here, before the next batch allocates its outputs
            for (auto* input : inputs) {
                if (input->get_users().empty() && !input->is_output()) {
                    p.remove_if_dangling(*input);
                }
            }
        }
    }
    remove_unused_constants(p);

    // propagate_constants is executed after compile_graph pass.
    // If some users become static due to propagated constants, they can end up without selected_impl.
    // Re-select implementation for affected nodes to avoid runtime _impl-nullptr validation failure.
    for (auto* node : reselection_targets) {
        try_reselect_impl_for_node(node);
    }
}

bool propagate_constants::has_non_const_user(program_node& node) {
    if (!node.is_constant()) {
        return true;
    }
    const auto& users = node.get_users();
    return std::any_of(users.begin(), users.end(), [](const auto& user) {
        return !user->is_constant();
    });
}

bool propagate_constants::is_read_at_inference(program_node& node) {
    return has_non_const_user(node) || node.is_output();
}

// Groups the computed constant nodes into connected subgraphs, each in processing order. A subgraph without a node that
// is read at inference (a non-constant user or a network output) is dropped; remove_unused_constants removes its nodes.
std::vector<std::vector<program_node*>> propagate_constants::find_constant_subgraphs(program& p) {
    std::vector<program_node*> constants;
    for (const auto& node : p.get_processing_order()) {
        if (is_computed_constant(*node)) {
            constants.push_back(node);
        }
    }
    for (auto* node : constants) {
        add_weights_rotation(p, *node);
    }

    std::unordered_map<program_node*, size_t> subgraph_of;
    size_t count = 0;
    for (const auto& node : p.get_processing_order()) {
        if (!is_computed_constant(*node) || subgraph_of.count(node) != 0) {
            continue;
        }
        subgraph_of[node] = count;
        std::queue<program_node*> queue;
        queue.push(node);
        while (!queue.empty()) {
            auto* n = queue.front();
            queue.pop();
            std::vector<program_node*> neighbours(n->get_users().begin(), n->get_users().end());
            for (const auto& dep : n->get_dependencies()) {
                neighbours.push_back(dep.first);
            }
            for (auto* m : neighbours) {
                if (is_computed_constant(*m) && subgraph_of.emplace(m, count).second) {
                    queue.push(m);
                }
            }
        }
        ++count;
    }

    std::vector<std::vector<program_node*>> subgraphs(count);
    std::vector<bool> is_read(count, false);
    for (const auto& node : p.get_processing_order()) {
        auto it = subgraph_of.find(node);
        if (it != subgraph_of.end()) {
            subgraphs[it->second].push_back(node);
            is_read[it->second] = is_read[it->second] || is_read_at_inference(*node);
        }
    }
    std::vector<std::vector<program_node*>> read_subgraphs;
    for (size_t i = 0; i < count; ++i) {
        if (is_read[i]) {
            read_subgraphs.push_back(std::move(subgraphs[i]));
        }
    }
    return read_subgraphs;
}

void propagate_constants::add_weights_rotation(program& p, program_node& node) {
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
            auto& rotate_node = p.get_or_create(rotate_prim);
            p.add_intermediate(rotate_node, node, 0);
            p.get_or_create(rotate_prim).recalc_output_layouts(false);
            GPU_DEBUG_LOG << "Added " << rotate_reorder_id << " for transposing weights before " << node.id() << std::endl;
        }
    }
#else
    (void)p;
    (void)node;
#endif  // ENABLE_ONEDNN_FOR_GPU
}

// compile_graph selected the implementations of the constant nodes but their kernels are built only in
// build_implementations, after this pass. Build them here in one batch, the same way.
void propagate_constants::build_kernels(const std::vector<std::vector<program_node*>>& subgraphs, kernels_cache& cache) {
    // The layouts that compile_graph left dynamic can be computed now, except where they depend on the values of an
    // input computed in the same subgraph (range with a computed stop): those are known only once the subgraph ran
    for (const auto& subgraph : subgraphs) {
        for (auto* node : subgraph) {
            if (node->is_dynamic()) {
                node->recalc_output_layouts(true);
            }
            try_reselect_impl_for_node(node);
        }
    }

    for (const auto& subgraph : subgraphs) {
        for (auto* node : subgraph) {
            if (auto* impl = node->get_selected_impl()) {
                auto params = node->get_kernel_impl_params();
                cache.add_kernels_source(*params, impl->get_kernels_source());
            }
        }
    }
    cache.build_all();
    for (const auto& subgraph : subgraphs) {
        for (auto* node : subgraph) {
            if (auto* impl = node->get_selected_impl()) {
                auto params = node->get_kernel_impl_params();
                impl->init_kernels(cache, *params);
                impl->reset_kernels_source();
            }
        }
    }
    cache.reset();
}

// Bytes of the outputs of all nodes of a subgraph, which are allocated while it runs; nothing when an output size is
// known only once the subgraph ran
std::optional<size_t> propagate_constants::allocation_size(const std::vector<program_node*>& subgraph) {
    size_t bytes = 0;
    for (auto* node : subgraph) {
        for (size_t i = 0; i < node->get_outputs_count(); ++i) {
            const auto& out_layout = node->get_output_layout(false, i);
            if (out_layout.is_dynamic()) {
                return std::nullopt;
            }
            bytes += out_layout.bytes_count();
        }
    }
    return bytes;
}

// Splits the subgraphs into ranges of consecutive ones that run together with one wait. A range allocates no more than
// the largest subgraph does on its own, so batching does not raise the peak memory of running them one by one. A
// subgraph whose size is not known before it runs is a range of its own.
std::vector<std::pair<size_t, size_t>> propagate_constants::make_batches(const std::vector<std::vector<program_node*>>& subgraphs) {
    std::vector<std::optional<size_t>> sizes;
    size_t limit = 0;
    for (const auto& subgraph : subgraphs) {
        sizes.push_back(allocation_size(subgraph));
        limit = std::max(limit, sizes.back().value_or(0));
    }

    std::vector<std::pair<size_t, size_t>> batches;
    size_t begin = 0;
    size_t bytes = 0;
    for (size_t s = 0; s < subgraphs.size(); ++s) {
        if (s > begin && (!sizes[s] || !sizes[s - 1] || bytes + *sizes[s] > limit)) {
            batches.emplace_back(begin, s);
            begin = s;
            bytes = 0;
        }
        bytes += sizes[s].value_or(0);
    }
    batches.emplace_back(begin, subgraphs.size());
    return batches;
}

// Runs the nodes of subgraphs [begin, end) with their compiled implementations and returns the memory of the nodes read
// at inference. All instances are removed again, so intermediates are freed when the batch is done.
std::vector<std::pair<program_node*, memory::ptr>> propagate_constants::execute_subgraphs(network& net,
                                                                                          const std::vector<program_node*>& inputs,
                                                                                          const std::vector<std::vector<program_node*>>& subgraphs,
                                                                                          size_t begin,
                                                                                          size_t end) {
    std::vector<primitive_id> added;
    for (auto* input : inputs) {
        net.add_detached_instance(*input);
        added.push_back(input->id());
    }
    std::vector<program_node*> nodes;
    std::vector<std::shared_ptr<primitive_inst>> insts;
    for (size_t s = begin; s < end; ++s) {
        for (auto* node : subgraphs[s]) {
            nodes.push_back(node);
            insts.push_back(net.add_detached_instance(*node));
            added.push_back(node->id());
        }
    }

    for (auto& inst : insts) {
        NODE_DEBUG(*inst);
        inst->prepare_primitive();
        // network::set_arguments binds the kernel arguments of static instances once before the first execution
        if (!inst->is_dynamic() && !inst->can_be_optimized() && inst->get_impl() != nullptr) {
            inst->set_arguments();
        }
        inst->execute();
    }
    net.get_stream().finish();

    std::vector<std::pair<program_node*, memory::ptr>> outputs;
    for (size_t i = 0; i < nodes.size(); ++i) {
        if (is_read_at_inference(*nodes[i])) {
            outputs.emplace_back(nodes[i], insts[i]->output_memory_ptr());
        }
    }
    insts.clear();
    for (const auto& id : added) {
        net.remove_detached_instance(id);
    }
    return outputs;
}

// A weights reorder that is the only user of its data input is recorded, so the weightless cache can replay it on import.
propagate_constants::cache_tuple propagate_constants::get_weightless_cache_info(program_node& node) {
    if (!node.is_type<reorder>() || node.get_dependencies().size() != 1) {
        return {};
    }
    auto& in = node.get_dependency(0);
    if (!in.is_type<data>() || in.get_users().size() != 1) {
        return {};
    }
    auto rprim = node.as<reorder>().get_primitive();
    return {in.as<data>().get_primitive()->cache_info, std::make_shared<layout>(in.get_output_layout()), std::make_shared<reorder>(*rprim)};
}

void propagate_constants::replace_with_data(program& p,
                                            program_node& curr_node,
                                            memory::ptr mem,
                                            const cache_tuple& cache_info,
                                            bool was_dynamic,
                                            std::unordered_set<program_node*>& reselection_targets) {
    auto cache_manager = std::get<0>(cache_info);
    auto in_layout = std::get<1>(cache_info);
    auto reorder_prim = std::get<2>(cache_info);

    auto const_data = std::make_shared<data>("_cldnn_const_prop_" + curr_node.id(),
                                             mem, /* <<< REMOVE ME WHEN POSSIBLE */
                                             cache_manager);
    auto& new_node = p.get_or_create(const_data);

    // Remove dependencies
    auto curr_node_deps = curr_node.get_dependencies();
    for (auto& dep : curr_node_deps) {
        auto dep_users = dep.first->get_users();
        for (auto& dep_user : dep_users) {
            if (dep_user == &curr_node) {
                program::remove_connection(*dep.first, curr_node);
            }
        }
    }

    if (in_layout && reorder_prim) {
        new_node.as<data>().get_primitive()->cache_info->apply_reorder(in_layout, reorder_prim);
    }

    curr_node.dependencies.clear();
    // Constant users are replaced by data or removed later. Disconnect them on both sides: p.replace destroys curr_node,
    // and a user that still lists it as a dependency would read freed memory when it is replaced itself
    std::vector<program_node*> const_users;
    for (auto* user : curr_node.users) {
        if (user->is_constant()) {
            const_users.push_back(user);
        }
    }
    for (auto* user : const_users) {
        program::remove_connection(curr_node, *user);
    }
    p.replace(curr_node, new_node);
    new_node.recalc_output_layout(was_dynamic);

    // Collect transitively affected user nodes when dynamic → static transition occurs.
    // Only users of constants that transitioned from dynamic to static need impl reselection.
    if (was_dynamic && !new_node.get_output_layout(false).is_dynamic()) {
        std::queue<program_node*> queue;
        for (const auto& user : new_node.get_users()) {
            queue.push(user);
        }
        while (!queue.empty()) {
            auto* n = queue.front();
            queue.pop();
            if (reselection_targets.count(n) > 0) {
                continue;
            }
            reselection_targets.insert(n);
            if (!n->is_all_valid_output_layouts()) {
                for (const auto& user : n->get_users()) {
                    queue.push(user);
                }
            }
        }
    }
}

// Disconnects a node from its dependencies and users and removes it from the program.
void propagate_constants::remove_node(program& p, program_node& node) {
    for (auto& dep : node.dependencies) {
        dep.first->users.remove(&node);
    }
    node.dependencies.clear();
    for (auto* usr : node.users) {
        auto& usr_deps = usr->dependencies;
        usr_deps.erase(std::remove_if(usr_deps.begin(),
                                      usr_deps.end(),
                                      [&](const std::pair<program_node*, int>& dep) {
                                          return &node == dep.first;
                                      }),
                       usr_deps.end());
    }
    node.users.clear();
    if (!node.is_output()) {
        auto rem = p.remove_if_dangling(node);
        assert(rem && "Non-output constant node which has only constant users should have been removed during constants "
                      "propagation pass");
        (void)rem;
    }
}

// remove all nodes which are no longer relevant, i.e. nodes which:
// 1. are constants, and
// 2. do not have non-const user (so their data are not used during inference), and
// 3. are not marked as outputs.
void propagate_constants::remove_unused_constants(program& p) {
    auto proc_itr = p.get_processing_order().begin();
    while (proc_itr != p.get_processing_order().end()) {
        const auto& node = (*proc_itr++);
        if (!node->is_constant()) {
            continue;
        }
        if (has_non_const_user(*node) || (node->is_output() && !node->is_type<data>())) {
            continue;
        }
        remove_node(p, *node);
    }
}
