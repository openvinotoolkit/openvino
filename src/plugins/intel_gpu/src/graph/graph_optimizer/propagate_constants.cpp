// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "data_inst.h"
#include "intel_gpu/graph/network.hpp"
#include "intel_gpu/graph/program.hpp"
#include "intel_gpu/runtime/debug_configuration.hpp"
#include "intel_gpu/runtime/engine.hpp"
#include "intel_gpu/runtime/internal_properties.hpp"
#include "intel_gpu/runtime/itt.hpp"
#include "mutable_data_inst.h"
#include "pass_manager.h"
#include "program_node.h"
#include "registry/implementation_manager.hpp"
#ifdef ENABLE_ONEDNN_FOR_GPU
#    include "graph/impls/onednn/utils.hpp"
#    include "reorder_inst.h"
#endif  // ENABLE_ONEDNN_FOR_GPU
#include <list>
#include <memory>
#include <queue>
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

}  // namespace

// ToDo remove friendship relation from  program_node and program
void propagate_constants::run(program& p) {
    OV_ITT_SCOPED_TASK(ov::intel_gpu::itt::domains::intel_gpu_plugin, "pass::PropagateConstants");
    for (const auto& node : p.get_processing_order()) {
        if (node->is_constant()) {
            handle_constant(p, *node);
        }
    }

    auto&& to_replace = _defer_materialization ? defer_to_import(p.get_engine())
                                               : calculate(p.get_engine(), p.get_config(), p.get_task_executor());

    // remove all nodes which are no longer relevant, i.e. nodes which:
    // 1. are constants, and
    // 2. do not have non-const user (so their data are not used during inference), and
    // 3. are not marked as outputs.
    // in case if node has either non-const user or is marked as output, it should be replace with cldnn::data rather
    // than removed (see next loop)
    auto proc_itr = p.get_processing_order().begin();
    while (proc_itr != p.get_processing_order().end()) {
        const auto& node = (*proc_itr++);
        if (!node->is_constant()) {
            continue;
        }
        if (has_non_const_user(*node) || (node->is_output() && !node->is_type<data>())) {
            continue;
        }

        auto& users = node->users;
        auto& deps = node->dependencies;

        for (size_t idx = 0; idx < deps.size(); idx++) {
            deps.at(idx).first->users.remove(node);
        }
        deps.clear();

        for (auto& usr : users) {
            auto& usr_deps = usr->dependencies;
            usr_deps.erase(std::remove_if(usr_deps.begin(),
                                          usr_deps.end(),
                                          [&](const std::pair<program_node*, int>& dep) {
                                              return node == dep.first;
                                          }),
                           usr_deps.end());
        }
        users.clear();

        if (!node->is_output()) {
            auto rem = p.remove_if_dangling(*node);
            assert(rem && "Non-output constant node which has only constant users should have been removed during constants "
                          "propagation pass");
            (void)rem;
        }
    }

    // replace all constant nodes which are relevant for inference (either used by non-const user or marked as output)
    // with recomputed cldnn::data
    std::unordered_set<program_node*> reselection_targets;
    for (auto& entry : to_replace) {
        auto& id_to_replace = std::get<0>(entry);
        auto mem_impl = std::get<1>(entry);
        auto cache_info = std::get<2>(entry);
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
                if (dep_user == &curr_node) {
                    p.remove_connection(*dep.first, curr_node);
                }
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
                                             [](program_node* node) {
                                                 return node->is_constant();
                                             }),
                              curr_node.users.end());
        bool was_dynamic = curr_node.get_output_layout().is_dynamic();
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

    // propagate_constants is executed after compile_graph pass.
    // If some users become static due to propagated constants, they can end up without selected_impl.
    // Re-select implementation for affected nodes to avoid runtime _impl-nullptr validation failure.
    for (auto* node : reselection_targets) {
        try_reselect_impl_for_node(node);
    }
}

bool propagate_constants::has_non_const_user(program_node& node) const {
    if (!node.is_constant()) {
        return true;
    }
    for (const auto& user : node.get_users()) {
        if (!user->is_constant()) {
            return true;
        }
    }
    return false;
}

// `data -> reorder` is replayable at load time on top of the original weights, so its value need not
// live in the blob. Weightless caching already does this; HW-free compile reuses the same recipe.
std::map<primitive_id, propagate_constants::cache_tuple> propagate_constants::collect_weightless_recipes() const {
    std::map<primitive_id, cache_tuple> weightless_cache_map;
    for (auto& cin : const_inputs) {
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
    return weightless_cache_map;
}

propagate_constants::replacement_list propagate_constants::calculate(engine& engine,
                                                                    const ExecutionConfig& config,
                                                                    std::shared_ptr<ov::threading::IStreamsExecutor> task_executor) {
    if (!has_non_trivial_constants) {
        return {};
    }

    ExecutionConfig cf_config = config.clone();
    cf_config.set_property(ov::intel_gpu::optimize_data(false));
    cf_config.set_property(ov::intel_gpu::custom_outputs(const_outputs));
    cf_config.finalize(engine);
    network::ptr net = network::build_network(engine, nodes, cf_config, task_executor, true);
    std::map<primitive_id, cache_tuple> weightless_cache_map = collect_weightless_recipes();
    for (auto& cin : const_inputs) {
        net->set_input_data(cin->id(), cin->get_attached_memory_ptr());
    }

    net->execute({});
    net->reset_execution(true);  // wait for computations to complete
    auto outputs = net->get_outputs();

    std::list<std::tuple<primitive_id, memory::ptr, cache_tuple>> ret;
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

// Walks back from a folded constant to the data nodes feeding it, capturing the primitives in
// between. Constant nodes only depend on constant nodes, so the result is always self-contained.
const_chain propagate_constants::build_chain(program_node& out) const {
    const_chain chain;
    chain.output_id = out.id();
    std::set<primitive_id> seen;

    std::function<bool(program_node&)> visit = [&](program_node& node) -> bool {
        if (!seen.insert(node.id()).second)
            return true;
        if (node.is_type<data>()) {
            auto mem = node.as<data>().get_attached_memory_ptr();
            if (mem == nullptr)
                return false;
            const_chain::leaf l{node.id(), node.get_output_layout(), std::vector<uint8_t>(mem->size())};
            mem_lock<uint8_t, mem_lock_type::read> lock(mem, mem->get_engine()->get_service_stream());
            std::copy_n(lock.data(), l.bytes.size(), l.bytes.begin());
            chain.leaves.push_back(std::move(l));
            return true;
        }
        if (!node.is_constant() || node.get_dependencies().empty())
            return false;
        for (const auto& dep : node.get_dependencies()) {
            if (!visit(*dep.first))
                return false;
        }
        // deps first keeps the order topological
        std::vector<input_info> ins;
        for (const auto& dep : node.get_dependencies())
            ins.emplace_back(dep.first->id(), dep.second);
        chain.prims.push_back(std::const_pointer_cast<primitive>(node.get_primitive()));
        chain.prim_inputs.push_back(std::move(ins));
        return true;
    };

    if (!visit(out) || chain.prims.empty() || chain.leaves.empty())
        return {};
    return chain;
}

propagate_constants::replacement_list propagate_constants::defer_to_import(engine& engine) {
    if (!has_non_trivial_constants) {
        return {};
    }

    auto weightless_cache_map = collect_weightless_recipes();

    auto find_node = [&](const primitive_id& id) -> program_node* {
        for (const auto& n : nodes) {
            if (n->id() == id)
                return n.get();
        }
        return nullptr;
    };

    replacement_list ret;
    std::vector<std::string> unsupported;
    std::set<primitive_id> visited;

    for (const auto& out_id : const_outputs) {
        if (!visited.insert(out_id).second)
            continue;

        auto* node = find_node(out_id);
        OPENVINO_ASSERT(node != nullptr, "[GPU] propagate_constants: constant output ", out_id, " is not in the chain set");

        // Describes the chain for the diagnostics below: the folded node plus its constant inputs.
        auto describe = [&]() {
            std::string s = out_id + " (" + node->get_primitive()->type_string() + " <- ";
            for (const auto& dep : node->get_dependencies())
                s += dep.first->get_primitive()->type_string() + std::string(" ");
            return s + ")";
        };

        auto it = weightless_cache_map.find(out_id);
        const bool has_recipe = it != weightless_cache_map.end() && std::get<0>(it->second) != nullptr;

        if (node->is_shape_infer_dep()) {
            // The host reads these bytes while inferring shapes, so a layout-only placeholder is not
            // enough. Needs compile-time evaluation (CPU) rather than deferral.
            unsupported.push_back(describe() + ": value is a shape inference dependency");
            continue;
        }
        if (!has_recipe || !std::get<0>(it->second)->is_managed()) {
            // Either not the single `data -> reorder` shape, or no reference to the original model.
            // Both are covered by carrying the whole constant sub-graph in the blob.
            auto chain = build_chain(*node);
            if (chain.prims.empty()) {
                unsupported.push_back(describe() + ": constant chain could not be captured");
                continue;
            }
            auto manager = std::make_shared<weightless_cache_manager>();
            manager->apply_chain(std::move(chain));
            const auto chain_layout = node->get_output_layout();
            auto chain_alloc = engine.get_preferred_memory_allocation_type(chain_layout.format.is_image_2d());
            ret.push_back({out_id, engine.allocate_memory(chain_layout, chain_alloc, false),
                           cache_tuple{manager, nullptr, nullptr}});
            continue;
        }
        // Layout comes from shape inference, which needs no GPU. The buffer stays empty by design:
        // data::save serializes the layout and the recipe, and import rebuilds the value.
        const auto out_layout = node->get_output_layout();
        auto alloc_type = engine.get_preferred_memory_allocation_type(out_layout.format.is_image_2d());
        ret.push_back({out_id, engine.allocate_memory(out_layout, alloc_type, false), it->second});
    }

    if (!unsupported.empty()) {
        std::string msg = "[GPU] HW-free compile cannot defer " + std::to_string(unsupported.size()) +
                          " constant chain(s) to import time:\n";
        for (const auto& u : unsupported)
            msg += "  - " + u + "\n";
        OPENVINO_THROW(msg);
    }

    return ret;
}

void propagate_constants::handle_constant(program& prog, program_node& node) {
    if (!node.is_type<data>()) {
        add_constant(prog, node);
        if (has_non_const_user(node)) {
            const_outputs.push_back(node.id());
        }
    }
}

void propagate_constants::add_constant(program& prog, program_node& node) {
    if (node.is_type<data>()) {
        return;
    }
    nodes.insert(prog.get_node_ptr(node.get_primitive()->id));
    has_non_trivial_constants = true;

    // if a node is either an endpoint or an output, always add it as an output
    if (node.is_endpoint() || node.is_output()) {
        const_outputs.push_back(node.id());
    }

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
            GPU_DEBUG_LOG << "Added " << rotate_reorder_id << " for transposing weights before " << node.id() << std::endl;
        }
    }
#endif  // ENABLE_ONEDNN_FOR_GPU
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
    for (const auto& dep : deps) {
        if (dep.first->is_type<data>()) {
            auto dep_ptr = prog.get_node_ptr(dep.first->get_primitive()->id);
            if (nodes.find(dep_ptr) == nodes.end()) {
                nodes.insert(dep_ptr);
                const_inputs.push_back(&dep.first->as<data>());
            }
        }
    }
}
