// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/graph/kernel_impl_params.hpp"
#include "intel_gpu/primitives/implementation_desc.hpp"
#include "intel_gpu/runtime/stream.hpp"
#include "program_helpers.h"
#include "primitive_inst.h"
#include "data_inst.h"
#include "mutable_data_inst.h"
#include "input_layout_inst.h"
#include "arg_max_min_inst.h"
#include "fully_connected_inst.h"
#include "convolution_inst.h"
#include "crop_inst.h"
#include "pooling_inst.h"
#include "permute_inst.h"
#include "resample_inst.h"
#include "reshape_inst.h"
#include "reorder_inst.h"
#include "eltwise_inst.h"
#include "loop_inst.h"
#include "deconvolution_inst.h"
#include "shape_of_inst.h"
#include "softmax_inst.h"
#include "strided_slice_inst.h"
#include "scatter_elements_update_inst.h"
#include "scatter_nd_update_inst.h"
#include "scatter_update_inst.h"
#include "gemm_inst.h"
#include "assign_inst.h"
#include "read_value_inst.h"
#include "kv_cache_inst.h"
#include "condition_inst.h"
#include "paged_attention_inst.h"
#include "gather_inst.h"
#include "broadcast_inst.h"
#include "dynamic_quantize_inst.h"
#include "swiglu_inst.h"
#include "experimental_detectron_roi_feature_extractor_inst.hpp"
#include "registry/implementation_manager.hpp"
#include "registry/registry.hpp"
#include "graph_optimizer/prepare_buffer_fusing.h"

#include "intel_gpu/plugin/common_utils.hpp"
#include "intel_gpu/plugin/multi_tensor_variable_state.hpp"
#include "intel_gpu/plugin/sync_infer_request.hpp"
#include "intel_gpu/graph/network.hpp"
#include "intel_gpu/graph/serialization/set_serializer.hpp"
#include "intel_gpu/runtime/engine.hpp"
#include "intel_gpu/runtime/memory.hpp"
#include "intel_gpu/runtime/debug_configuration.hpp"
#include "intel_gpu/runtime/compilation_context.hpp"

#include "json_object.h"
#include <string>
#include <vector>
#include <memory>
#include <algorithm>

#ifdef ENABLE_ONEDNN_FOR_GPU
#include <impls/onednn/utils.hpp>
#endif

namespace cldnn {
namespace {

template <typename T>
bool is_optimized_output_user(const T user) {
    if (user->can_be_optimized()) {
        if (user->is_output())
            return true;

        auto users = user->get_users();
        for (const auto& u : users) {
            if (is_optimized_output_user(u)) {
                return true;
            }
        }
        return false;
    }
    return false;
}
bool is_output_buffer(const primitive_inst* prim, bool runtime_alloc) {
    if (prim->is_output())
        return true;

    // Try to recursively find any optimized out user which is also network output
    if (runtime_alloc) {
        // Try to recursively find any optimized out user which is also network output
        for (const auto& user : prim->get_user_insts()) {
            if (is_optimized_output_user<const primitive_inst*>(user)) {
                return true;
            }
        }
    } else {
        for (const auto& user : prim->get_node().get_users()) {
            if (is_optimized_output_user<const program_node*>(user)) {
                return true;
            }
        }
    }
    return false;
}

bool is_user_cpu(const program_node* user) {
    if (user->can_be_optimized()) {
        auto users = user->get_users();
        for (const auto& u : users) {
            if (is_user_cpu(u)) {
                return true;
            }
        }
    }
    bool is_cpu = user->get_selected_impl() ? user->get_selected_impl()->is_cpu()
                                            : user->get_preferred_impl_type() == impl_types::cpu;
    return is_cpu;
}

bool has_cpu_user_not_shape_of(const program_node* user) {
    if (user->can_be_optimized()) {
        auto users = user->get_users();
        for (const auto& u : users) {
            if (has_cpu_user_not_shape_of(u)) {
                return true;
            }
        }
        return false;
    }
    if (auto impl = user->get_selected_impl())
        return impl->is_cpu() && !user->is_type<shape_of>();
    return false;
}

bool has_any_cpu_user_not_shape_of(const std::list<const program_node*>& users) {
    for (const auto& user : users) {
        if (has_cpu_user_not_shape_of(user))
            return true;
    }
    return false;
}
}  // namespace

bool is_any_user_cpu(const std::list<const program_node*>& users) {
    for (const auto& user : users) {
        if (is_user_cpu(user))
            return true;
    }
    return false;
}

static memory::ptr get_memory_from_pool(engine& _engine,
                                uint32_t net_id,
                                memory_pool& pool,
                                const program_node& _node,
                                const layout& layout,
                                allocation_type type,
                                bool reusable_across_network,
                                const std::unordered_set<size_t>& memory_dependencies,
                                bool reset = true,
                                memory* curr_memory = nullptr) {
    OPENVINO_ASSERT(!layout.is_dynamic() || layout.has_upper_bound(),
                    "[GPU] Can't allocate output for dynamic layout without upper bound");
    // Use layout with max tensor for dynamic shape with upper bound
    if (_node.get_program().get_config().get_enable_memory_pool()) {
        if (curr_memory != nullptr)
            pool.release_memory(curr_memory, _node.get_unique_id(), _node.id(), net_id);
        return pool.get_memory(layout,
                               _node.id(),
                               _node.get_unique_id(),
                               net_id,
                               memory_dependencies,
                               type,
                               reusable_across_network,
                               reset,
                               _node.is_dynamic());
    }
    return pool.get_memory(layout, type, reset);
}

std::shared_ptr<kernel_impl_params> primitive_impl::get_weights_reorder_kernel_params() const {
    if (!need_weights_reorder())
        return nullptr;

    auto reorder_kernel_params = std::make_shared<kernel_impl_params>();
    auto prim = std::make_shared<reorder>("", input_info(), _weights_reorder_params);
    reorder_kernel_params->desc = prim;
    reorder_kernel_params->unique_id = _weights_reorder_params->hash();
    reorder_kernel_params->input_layouts.push_back(_weights_reorder_params->get_input_layout());
    reorder_kernel_params->output_layouts.push_back(_weights_reorder_params->get_output_layout());
    return reorder_kernel_params;
}

kernel_impl_params primitive_impl::static_canonicalize_shapes(const kernel_impl_params& impl_params) {
    auto updated_impl_params = canonicalize_fused_shapes(impl_params);

    for (auto& input_layout : updated_impl_params.input_layouts) {
        input_layout.set_partial_shape(extend_shape_to_rank_from_end(input_layout.get_partial_shape()));
    }

    for (auto& output_layout : updated_impl_params.output_layouts) {
        output_layout.set_partial_shape(extend_shape_to_rank_from_end(output_layout.get_partial_shape()));
    }

    return updated_impl_params;
}

uint32_t primitive_inst::get_network_id() const { return _network.get_id(); }

void primitive_inst::check_memory_to_set(const memory& mem, const layout& l) const {
    // The layout with empty tensor (scalar) is regarded as 1 dimension with value 1
    bool single_value_layout = false;
    if (!l.is_dynamic()) {
        const auto& layout_ps = l.get_partial_shape();
        single_value_layout = (layout_ps.size() == 1 && layout_ps[0] == 1);
    }

    const auto& mem_layout = mem.get_layout();
    OPENVINO_ASSERT((mem_layout == l)
                    || l.is_dynamic()
                    || (mem_layout.get_partial_shape().size() == 0 && single_value_layout),
                    "[GPU] Unexpected layout of input memory for ", id(), " node!\n",
                    "Node layout: ", l.to_short_string(), "\n",
                    "Memory layout: ", mem_layout.to_short_string());

    // check shared image/buffer compatibility, if applicable
    auto params = mem.get_internal_params();
    if (params.mem_type != shared_mem_type::shared_mem_empty) {
        auto& net_engine = get_network().get_engine();
        auto& mem_engine = *mem.get_engine();
        OPENVINO_ASSERT(mem.is_allocated_by(net_engine), "[GPU] Can't set memory due to engines mismatch. ",
                        "Network was created for ", &net_engine, " (",
                        net_engine.get_device_info().dev_name, ") engine",
                        " while memory object was allocated for ", &mem_engine, " (",
                        mem_engine.get_device_info().dev_name, ")");

        switch (params.mem_type) {
        case shared_mem_type::shared_mem_vasurface:
        case shared_mem_type::shared_mem_image:
            OPENVINO_ASSERT(l.format.is_image_2d(), "Attempt to set user-supplied input or output image instead of a buffer");
            break;
        case shared_mem_type::shared_mem_buffer:
        case shared_mem_type::shared_mem_dxbuffer:
            OPENVINO_ASSERT(!l.format.is_image_2d(), "Attempt to set user-supplied input or output buffer instead of an image");
            break;
        case shared_mem_type::shared_mem_usm:
            break;
        default:
            OPENVINO_THROW("Attempt to set user-supplied input or output memory of unknown/invalid type");
            break;
        }
    }
}

event::ptr primitive_inst::set_output_memory(memory::ptr mem_new, bool check, size_t idx) {
    auto& eng = _network.get_engine();
    // skip all the buzz if no action actually required
    event::ptr ev = nullptr;
    if (_outputs[idx] && eng.is_the_same_buffer(*mem_new, *_outputs[idx])) {
        return nullptr;
    }

    const auto& ol = _impl_params->get_output_layout(idx);

    if (check)
        check_memory_to_set(*mem_new, ol);

    if (is_constant()) {
        ev = mem_new->copy_from(_network.get_stream(), *_outputs[idx], false);
    } else {
        _outputs[idx] = mem_new;
        _max_output_layout_count[idx] = mem_new->get_layout().get_linear_size();
    }
    return ev;
}

void primitive_inst::update_shape() {
    OV_ITT_SCOPED_TASK(ov::intel_gpu::itt::domains::intel_gpu_plugin, openvino::itt::handle("update_shape: " + id()));
    GPU_DEBUG_PROFILED_STAGE(instrumentation::pipeline_stage::shape_inference);
    if (update_shape_done_by_other) {
        update_shape_done_by_other = false; // reset
        GPU_DEBUG_TRACE_DETAIL << id() << ": update shape is done by other: "
                               << _impl_params->output_layouts[0].to_short_string() << std::endl;
        return;
    }
    bool input_shape_changed = false;
    for (size_t i = 0; i < _deps.size(); i++) {
        auto idx = _deps[i].second;
        auto new_shape = _deps[i].first->_impl_params->get_output_layout(idx);
        if (_impl_params->get_input_layout(i) != new_shape) {
            GPU_DEBUG_TRACE_DETAIL << id() << ": update shape dep [" << i << "] : " << _deps[i].first->id()
                                   << " was: " << _impl_params->get_input_layout(i).to_short_string()
                                   << " now: " << new_shape.to_short_string() << std::endl;
            _impl_params->input_layouts[i] = new_shape;
            input_shape_changed = true;
        }
    }
    if (get_node().is_type<read_value>()) {
        auto prim = get_node().as<read_value>().get_primitive();
        const auto& variable_id = prim->variable_id;
        auto& variable = get_network().get_variable(variable_id);

        auto update_state_layout = [&](ov::intel_gpu::VariableStateBase& variable, layout new_layout, size_t layout_idx) {
            // If variable is not set and we have an initializer - use it's shape as shape of variable
            if (!variable.is_set() && _impl_params->input_layouts.size() > layout_idx) {
                new_layout = _impl_params->get_input_layout(layout_idx);
            }

            // If we still have a dynamic dimension, which basiclly means that we don't have an initializer, then replace dynamic dims with 0
            if (new_layout.is_dynamic()) {
                auto pshape = new_layout.get_partial_shape();
                for (auto& d : pshape) {
                    if (d.is_dynamic()) {
                        d = 0;
                    }
                }
                new_layout.set_partial_shape(pshape);
            }

            variable.set_layout(new_layout);

            if (_impl_params->state_layouts[layout_idx] != new_layout) {
                _impl_params->state_layouts[layout_idx] = new_layout;
                GPU_DEBUG_TRACE_DETAIL << "Update " << layout_idx << " layout: " << new_layout.to_short_string() << "\n";
                input_shape_changed = true;
            }
        };

        if (_impl_params->state_layouts.empty())
            _impl_params->state_layouts.resize(1);

        // Initial variable shape is taken from variable itself
        auto new_layout = variable.get_layout();
        update_state_layout(variable, new_layout, 0);

        if (prim->num_outputs > 1) {
            if (auto compressed_cache_variable = dynamic_cast<ov::intel_gpu::VariableStateIndirectKVCacheCompressed*>(&variable)) {
                _impl_params->state_layouts.resize(compressed_cache_variable->has_zp_state() ? 3 : 2);

                auto scales_state = compressed_cache_variable->get_compression_scale_state();
                auto new_scales_layout = scales_state->get_layout();
                update_state_layout(*scales_state, new_scales_layout, 1);

                if (compressed_cache_variable->has_zp_state()) {
                    auto zp_state = compressed_cache_variable->get_compression_zp_state();
                    auto new_zp_layout = zp_state->get_layout();
                    update_state_layout(*zp_state, new_zp_layout, 2);
                }
            }
        }
    }

    set_flag(ExecutionFlags::SHAPE_CHANGED, input_shape_changed);

    // We assume that tensor ranks are static, thus shape_of doesn't need to update anything even if input shape is dynamic
    if (_node->is_type<shape_of>() && !input_shape_changed) {
        return;
    }

    // if input shape is not changed, loop doesn't need to update anything.
    // because actual output layout will be calculated after the end of body network execution.
    if (_node->is_type<loop>() && !input_shape_changed) {
        return;
    }

    // Do not update shapes in shape_of subraph if shape_of's input shape is not changed
    if (_node->is_in_shape_of_subgraph()) {
        bool subgraph_input_changed = false;
        for (size_t i = 0; i < dependant_shape_of_insts.size(); i++) {
            if (dependant_shape_of_insts[i]->get_flag(ExecutionFlags::SHAPE_CHANGED)) {
                subgraph_input_changed = true;
                break;
            }
        }
        if (!subgraph_input_changed) {
            GPU_DEBUG_TRACE_DETAIL << id() << ": skip shape_update, because it is in shape_of_subgraph and input shape is not changed\n";
            unset_flag(ExecutionFlags::SHAPE_CHANGED);
            return;
        }
    }

    // Even though the predecessors' shapes are not changed, the output shape might be updated by the mem_dep
    auto memory_deps = _node->get_const_memory_deps();
    for (auto& i : _node->get_shape_infer_dependencies()) {
        if (memory_deps.count(i) > 0) {
            continue;
        }
        if (i >= _deps.size())
            continue;

        if (_deps[i].first->get_node().is_in_shape_of_subgraph() &&
        (_deps[i].first->get_node().get_selected_impl() ? _deps[i].first->get_node().get_selected_impl()->is_cpu()
        : _deps[i].first->get_node().get_preferred_impl_type() == impl_types::cpu)) {
            bool can_skip = true;
            const auto& insts = _deps[i].first->dependant_shape_of_insts;
            for (auto& inst : insts) {
                can_skip &= !inst->get_flag(ExecutionFlags::SHAPE_CHANGED);
            }
            if (can_skip)
                continue;
        }

        input_shape_changed = true;
    }

    if (!_node->is_type<kv_cache>() && !input_shape_changed && _impl_params->get_output_layout().is_static())
        return;

    std::vector<event::ptr> dependencies_events;
    auto queue_type = get_network().get_stream().get_queue_type();
    bool has_runtime_deps = false;
    for (auto& i : _node->get_shape_infer_dependencies()) {
        // Some primitives may have flexible count of deps (e.g. reshape), thus allow skipping some deps
        if (memory_deps.count(i) > 0 || i >= _node->get_dependencies().size()) {
            continue;
        }

        auto& dep = _deps[i].first;
        auto& dep_port = _deps[i].second;
        // exclude fused node from memory_deps
        if (_node->is_fused_dep(i)) {
            break;
        }

        auto dep_mem = dep->output_memory_ptr(dep_port);
        memory_deps.insert({i, dep_mem});

        // Ignore shape infer dependency for input_layout when processing paged_attention nodes
        if (get_node().is_type<paged_attention>() && dep->get_node().is_type<input_layout>()) {
            continue;
        }

        if (!get_node().is_type<shape_of>() &&
        !(dep->get_node().get_selected_impl() ? dep->get_node().get_selected_impl()->is_cpu() : dep->get_node().get_preferred_impl_type() == impl_types::cpu)) {
            has_runtime_deps = true;

            // Events may be not created for in-order queue, so take them for OOO queue only
            if (queue_type == QueueTypes::out_of_order && dep->get_impl_params()->out_event) {
                dependencies_events.push_back(dep->get_impl_params()->out_event);

                GPU_DEBUG_TRACE_DETAIL << id() << ": shape infer waits for " << i << " dependency\n";
            }
        }
    }

    if (has_runtime_deps) {
        OV_ITT_SCOPED_TASK(ov::intel_gpu::itt::domains::intel_gpu_plugin, openvino::itt::handle("update_shape_sync: " + id()));
        GPU_DEBUG_TRACE_DETAIL << "runtime synchronization for " << id() << " shape inference\n";
        if (!dependencies_events.empty() && queue_type == QueueTypes::out_of_order) {
            _network.get_stream().wait_for_events(dependencies_events);
        } else if (queue_type == QueueTypes::in_order) {
            _network.get_stream().finish();
        }
    }

    _impl_params->memory_deps = memory_deps;


    auto new_layouts = _node->type()->calc_output_layouts(*_node, *_impl_params);
    for (size_t idx = 0; idx != new_layouts.size(); ++idx) {
        auto& new_layout = new_layouts[idx];
        if (!_node->is_type<reshape>() || (!_node->get_input_layout(0).data_padding.is_dynamic() && !_node->can_be_optimized())) {
            auto data_padding = padding::max(_impl_params->get_output_layout(idx).data_padding, new_layout.data_padding);
            new_layout.data_padding = padding::max(_node->get_primitive()->get_output_padding(idx), data_padding);
        }

        if (_impl_params->get_output_layout(idx) != new_layout) {
            GPU_DEBUG_TRACE_DETAIL << id() << ": update shape: was: " << _impl_params->get_output_layout(idx).to_short_string()
                                    << " now: " << new_layout.to_short_string() << std::endl;
            set_flag(ExecutionFlags::SHAPE_CHANGED);
        }

        _impl_params->output_layouts[idx].data_padding = new_layout.data_padding;
        _impl_params->output_layouts[idx].set_partial_shape(new_layouts[idx].get_partial_shape());
    }

    // Update descriptors of fused operations and set output_layout's shape to all fused ops
    // It's legal as long as fused ops don't change the shape
    for (auto& fused_prim : _impl_params->fused_desc) {
        fused_prim.output_layout.set_partial_shape(_impl_params->get_output_layout().get_partial_shape());
    }

    if (get_node().is_type<assign>()) {
        auto desc = get_node().as<assign>().get_primitive();
        get_network().get_variable(desc->variable_id).set_layout(_impl_params->get_output_layout());
    }

    if (get_node().is_type<read_value>()) {
        auto desc = get_node().as<read_value>().get_primitive();
        auto& variable = get_network().get_variable(desc->variable_id);
        // Custom output layout update as update_output_layout handles paddings incorrectly for optimized out read_value + kv_cache pattern
        _impl_params->output_layouts[0] = variable.get_layout();

        if (auto compressed_cache_variable = dynamic_cast<ov::intel_gpu::VariableStateIndirectKVCacheCompressed*>(&variable)) {
            _impl_params->output_layouts[1] = compressed_cache_variable->get_compression_scale_state()->get_layout();

            if (compressed_cache_variable->has_zp_state()) {
               _impl_params->output_layouts[2] = compressed_cache_variable->get_compression_zp_state()->get_layout();
            }
        }
    }

    if (get_node().is_type<kv_cache>()) {
        auto desc = get_node().as<kv_cache>().get_primitive();
        auto var_mem_size = get_network().get_variable(desc->variable_info.variable_id).get_actual_mem_size();
        // Need to trigger realloc_if_needed
        if (var_mem_size < _impl_params->get_output_layout(0).get_linear_size())
            set_flag(ExecutionFlags::SHAPE_CHANGED);
    }
}

kernel_impl_params primitive_inst::get_fake_aligned_params_if_possible(kernel_impl_params const& orig_impl_param) {
    auto updated_params = _node->type()->get_fake_aligned_params(orig_impl_param);

    const auto &dev_info = get_node().get_program().get_engine().get_device_info();

    // The target HW of this patch is limited because of performance concern
    if ((dev_info.supports_immad && dev_info.dev_type == device_type::integrated_gpu) || dev_info.gfx_ver.major >= 20) {
        // Check whether the input node has enough space for output data. Otherwise, fake alignment is not possible due to page fault
        // i.e. predecessor node was supposed be increased already
        if (get_node().is_type<fully_connected>() && dependencies().size() > 0 && dep_memory(0).get_layout().is_static()
            && dep_memory(0).count() < updated_params.input_layouts[0].count()) {
            GPU_DEBUG_TRACE_DETAIL << "Roll back fake_aligned params for " << id()
                << "  allocated: " << dep_memory(0).count()
                << "  required: " << updated_params.input_layouts[0].count()
                << std::endl;
            updated_params = *_impl_params;
        }
    }
    return updated_params;
}

// Check if all dependencies and its predecessors are CPU or constant
static bool check_all_deps_cpu(const primitive_inst* inst) {
    return std::all_of(inst->dependencies().begin(), inst->dependencies().end(),
        [&](const std::pair<const primitive_inst*, int32_t>& dep) {
            if (dep.first->is_constant() ||
                (dep.first->get_impl() != nullptr && dep.first->get_impl()->is_cpu())) {
                return true;
            }
            // Check if the dependency can be optimized
            if (dep.first->can_be_optimized()) {
                return check_all_deps_cpu(dep.first);
            }
            return false;
        });
}

bool primitive_inst::all_dependencies_cpu_impl() const {
    return check_all_deps_cpu(this);
}

void primitive_inst::clear_output_memory() {
    _outputs[0] = nullptr;
    _max_output_layout_count[0] = 0;
}

void primitive_inst::realloc_if_needed(bool prev_execution_skipped) {
    OV_ITT_SCOPED_TASK(ov::intel_gpu::itt::domains::intel_gpu_plugin, openvino::itt::handle("realloc_if_needed: " + id()));
    GPU_DEBUG_PROFILED_STAGE(instrumentation::pipeline_stage::memory_allocation);

    const auto& users = get_user_insts();
    if (users.size() == 1 && users.front()->get_node().is_type<concatenation>() && users.front()->get_node().is_runtime_skippable()) {
        auto concat_inst = users.front();
        if (concat_inst->can_be_optimized()) {
            if (!concat_inst->allocation_done_by_other) {
                concat_inst->realloc_if_needed();
                concat_inst->allocation_done_by_other = true;
            }
            this->_outputs[0] = concat_inst->_outputs[0];
            GPU_DEBUG_TRACE_DETAIL << id() << ": use concat user's memory " << this->_outputs[0]->buffer_ptr() << std::endl;
            return;
        }
    }

    // Update param if fake_alignment is available
    auto updated_params = get_fake_aligned_params_if_possible(*_impl_params);

    const auto& actual_layouts = updated_params.output_layouts;
    OPENVINO_ASSERT(actual_layouts[0].is_static(), "[GPU] Can't realloc mem for dynamic layout");

    if (users.size() == 1 && users.front()->get_node().is_type<reorder>() && users.front()->can_be_optimized()) {
        auto reorder_inst = users.front();
        if (reorder_inst->is_output()
            && reorder_inst->output_memory_ptr()
            && get_network().has_output_remote_memory_ptr(reorder_inst->id())
            && get_network().get_engine().is_the_same_buffer(get_network().get_output_remote_memory(reorder_inst->id()), reorder_inst->output_memory())) {
            if (actual_layouts[0].get_linear_size() <= reorder_inst->get_max_output_layout_count()) {
                this->_outputs[0] = reorder_inst->_outputs[0];
                GPU_DEBUG_TRACE_DETAIL << id() << ": use reorder user's remote tensor memory " << this->_outputs[0]->buffer_ptr() << std::endl;
                return;
            } else {
                GPU_DEBUG_TRACE_DETAIL << reorder_inst->id() << " cannot be optimized for the mismatch between input layout and output layout" << std::endl;
                reorder_inst->set_can_be_optimized(false);
            }
        }
    }

    // input_layout node is supposed to always use external memory in dynamic case
    if (_node->is_type<input_layout>())
        return;

    auto& sp = *get_network().get_shape_predictor();
    std::vector<size_t> dt_sizes_in_B;
    for (size_t i = 0; i < actual_layouts.size(); ++i) {
        dt_sizes_in_B.push_back(ov::element::Type(actual_layouts[i].data_type).size());
    }
    // read_value/assign nodes are supposed to always use variable memory
    if (auto stateful_prim = dynamic_cast<memory_state::variable*>(this)) {
        auto& variable_id = stateful_prim->variable_id();
        auto& variable = get_network().get_variable(variable_id);
        if (_node->is_type<kv_cache>()) {
            // Reuse state memory as output for kv cache if possible
            // otherwise clear _outputs for the cases when mem was reused previously
            if (_impl_params->can_be_optimized()) {
                GPU_DEBUG_TRACE_DETAIL << id() << " : realloc_if_needed: Set kvcache output memory as variable memory " << variable.get_memory()->buffer_ptr()
                                    << " (ptr: " << variable.get_memory()->buffer_ptr()
                                    << ", actual_size: " << variable.get_actual_mem_size()/8 << " bytes"
                                    << ", variable layout " << variable.get_layout().to_short_string() << ")" << std::endl;

                _outputs[0] = variable.get_memory();

                if (auto compressed_cache_variable = dynamic_cast<ov::intel_gpu::VariableStateIndirectKVCacheCompressed*>(&variable)) {
                    _outputs[2] = compressed_cache_variable->get_compression_scale_state()->get_memory();

                    if (compressed_cache_variable->has_zp_state()) {
                        _outputs[3] = compressed_cache_variable->get_compression_zp_state()->get_memory();
                    }
                }

                // To record shape predictor
                for (size_t j = 0; j < _impl_params->output_layouts.size(); ++j)
                    sp.predict_preallocation_shape(id(), _impl_params->output_layouts[j], true, j);
                GPU_DEBUG_PROFILED_STAGE_MEMALLOC_INFO("can_be_optimized");
                return;
            } else if (_outputs[0] && variable.get_memory() && get_network().get_engine().is_the_same_buffer(*_outputs[0], *variable.get_memory())) {
                GPU_DEBUG_TRACE_DETAIL << id() << " : realloc_if_needed: Reset output mem" << std::endl;
                for (size_t j = 0; j < _impl_params->output_layouts.size(); ++j) {
                    _outputs[j] = nullptr;
                    _max_output_layout_count[j] = 0;
                }
            } else {
                GPU_DEBUG_TRACE_DETAIL
                    << id() << " : realloc_if_needed: can_be_optimized = false and memories are not being shared"
                    << std::endl;
                if (!get_network().is_reuse_variable_mem()) {
                    GPU_DEBUG_TRACE_DETAIL << "Update output mem with new variable mem" << std::endl;
                    _outputs[0] = variable.get_memory();
                    _max_output_layout_count[0] = variable.get_actual_mem_size() / dt_sizes_in_B[0];

                    if (auto compressed_cache_variable = dynamic_cast<ov::intel_gpu::VariableStateIndirectKVCacheCompressed*>(&variable)) {
                        _outputs[2] = compressed_cache_variable->get_compression_scale_state()->get_memory();

                        if (compressed_cache_variable->has_zp_state()) {
                            _outputs[3] = compressed_cache_variable->get_compression_zp_state()->get_memory();
                        }
                    }
                } else {
                    GPU_DEBUG_TRACE_DETAIL << "Can reuse variable mem of prev request" << std::endl;
                }
            }
        } else {
            variable.set_layout(_impl_params->output_layouts[0]);
            GPU_DEBUG_TRACE_DETAIL << id() << ": Update variable (ptr: " << variable.get_memory()->buffer_ptr()
                                   << ", actual_size:" << variable.get_actual_mem_size() << " bytes"
                                   << ", variable layout:" << variable.get_layout().to_short_string() << ")" << std::endl;

            if (auto compressed_cache_variable = dynamic_cast<ov::intel_gpu::VariableStateIndirectKVCacheCompressed*>(&variable)) {
                compressed_cache_variable->get_compression_scale_state()->set_layout(_impl_params->output_layouts[1]);

                if (compressed_cache_variable->has_zp_state()) {
                    compressed_cache_variable->get_compression_zp_state()->set_layout(_impl_params->output_layouts[2]);
                }
            }
        }
        // For nodes that can be optimized, variable memory is used as output memory
        // so there is no need for output memory reallocation
        if (can_be_optimized()) {
            _max_output_layout_count[0] = variable.get_actual_mem_size() / dt_sizes_in_B[0];

            if (auto compressed_cache_variable = dynamic_cast<ov::intel_gpu::VariableStateIndirectKVCacheCompressed*>(&variable)) {
                const size_t scale_idx = _node->is_type<read_value>() ? 1 : 2; // kv_cache or read_value
                _max_output_layout_count[scale_idx] = compressed_cache_variable->get_compression_scale_state()->get_actual_mem_size() / dt_sizes_in_B[1];
                if (compressed_cache_variable->has_zp_state()) {
                    _max_output_layout_count[scale_idx + 1] = compressed_cache_variable->get_compression_zp_state()->get_actual_mem_size() / dt_sizes_in_B[2];
                }
            }
            GPU_DEBUG_PROFILED_STAGE_MEMALLOC_INFO("can_be_optimized");
            return;
        }
    }

    // Update output layout with respect to FC's fake alignment
    auto updated_layouts = actual_layouts;
    std::vector<cldnn::primitive_inst *> user_insts;
    {
        const auto& user_insts_origin = get_user_insts();
        for (auto& user : user_insts_origin) {
            auto uid = user->id();
            if (user->get_node().is_type<fully_connected>() && user->is_dynamic() && user->_deps[0].first == this
                && std::find_if(user_insts_origin.begin(), user_insts_origin.end(), [&](cldnn::primitive_inst * uu){
                    for (auto dep_inst : uu->_deps) {
                        if (dep_inst.first->id() == uid)
                            return true;
                    }
                    return false;
                }) != user_insts_origin.end()) {
                    user_insts.insert(user_insts.begin(), user);
            } else {
                user_insts.push_back(user);
            }
        }
        OPENVINO_ASSERT(user_insts.size() == user_insts_origin.size(), "Should have same size between ",
                        user_insts.size(), " and ", user_insts_origin.size());
    }
    for (auto user : user_insts) {
        auto is_fused_prim_of_user = [&](primitive_id id) -> bool {
            for (auto& p : user->get_node().get_fused_primitives()) {
                if (p.has_outer_dep()) {
                    const auto start_idx = p.outer_dep_start_idx;
                    // exclude fused_node from total_num_deps
                    const auto end_idx = p.outer_dep_start_idx + p.total_num_deps -1;
                    for (size_t idx = start_idx; idx < end_idx; idx++) {
                        if (user->get_node().get_dependency(idx).id() == id) {
                            return true;
                        }
                    }
                }
            }
            return false;
        };
        // Since fake alignment is applicable for input tensor as well, make sure we allocate enough memory
        // to prevent reading beyond the allocated memory bounds
        if (user->get_node().is_type<fully_connected>() && user->is_dynamic()) {
            if (user->_deps[0].first == this || (is_fused_prim_of_user(id()) && user->update_shape_done_by_other)) {
                size_t dep_idx = 0;
                for (const auto& dep : user->_deps) {
                    if (dep.first->id() == id()) {
                        dep_idx = dep.second;
                        break;
                    }
                }
                GPU_DEBUG_TRACE_DETAIL << id() <<"'s " << dep_idx << "-th output is " << user->id() << "'s input" << std::endl;
                GPU_DEBUG_TRACE_DETAIL << "Check fc user " << user->id() << "'s fake alignment-ed input size" << std::endl;
                // Setting update_shape_done_by_other to false before running update_shape,
                // since update_Shape is already called in realloc_if_needed of current node's dep node
                // but current node's output layout is not updated to the this user node yet.
                user->update_shape_done_by_other = false;
                bool prev_shape_changed = user->get_flag(ExecutionFlags::SHAPE_CHANGED);
                user->update_shape();
                // Set again shape_change status if shape is changed in the prev udpate_shape() for this user node.
                if (prev_shape_changed)
                    user->set_flag(ExecutionFlags::SHAPE_CHANGED);
                user->update_shape_done_by_other = true;
                auto fc_impl_params = *user->_impl_params;
                auto fc_input_layout = user->get_node().type()->get_fake_aligned_params(fc_impl_params).input_layouts[0];
                if (fc_input_layout.bytes_count() > updated_layouts[dep_idx].bytes_count()) {
                    GPU_DEBUG_TRACE_DETAIL << id() << ": increase output layout allocation size from "
                                        << actual_layouts[dep_idx].to_short_string() << " -> "
                                        << fc_input_layout.to_short_string() << " to meet the input buffer alignment requirements for FC\n";
                    updated_layouts[dep_idx] = fc_input_layout;
                }

                // dynamic quantization is only applied to activation of FC
                if (get_node().is_type<dynamic_quantize>()) {
                    const auto& desc = get_node().as<dynamic_quantize>().get_primitive();
                    auto dyn_quan_scale_layout =
                        dynamic_quantize_inst::__calc_output_layouts<ov::PartialShape>(updated_layouts[dep_idx],
                                                                                       desc->attrs);
                    GPU_DEBUG_TRACE_DETAIL << "update layout of dynamic quantize scale parameter layout "
                                        << dyn_quan_scale_layout[1].to_short_string() << std::endl;
                    updated_params.output_layouts[1] = dyn_quan_scale_layout[1];
                }
            }
        }
    }

    // Clear out memory if was previously reused, but now primitive can't be optimized
    if (!_node->is_type<concatenation>() && (_node->is_runtime_skippable() || _node->is_type<crop>())) {
        std::function<void(cldnn::primitive_inst*, cldnn::memory::ptr)> reset_user_output_memory
                            = [&](cldnn::primitive_inst* curr_inst, cldnn::memory::ptr target_mem_ptr) {
            for (auto& user_inst : curr_inst->get_user_insts()) {
                auto curr_output_memory_ptr = user_inst->output_memory_ptr(0);
                if (user_inst->can_be_optimized()
                        && (curr_output_memory_ptr
                            && get_network().get_engine().is_the_same_buffer(*curr_output_memory_ptr, *target_mem_ptr))) {
                    user_inst->clear_output_memory();
                    reset_user_output_memory(user_inst, target_mem_ptr);
                }
            }
        };
        if (can_be_optimized()) {
            _max_output_layout_count = _deps[0].first->_max_output_layout_count;
            GPU_DEBUG_PROFILED_STAGE_MEMALLOC_INFO("can_be_optimized");
            // If the inst is optimized out but it executed at the previous iteration,
            // reset all output memory of users which was optimized out at the previous iteration.
            // Ex.
            // * iter0: node1(executed) -> node2(skipped) -> node3(skipped)
            // * iter1: node1(skipped)  -> node2(skipped) -> node3(executed)
            if (_outputs[0] && dep_memory_ptr(0)
                && !_network.get_engine().is_the_same_buffer(dep_memory(0), output_memory(0))) {
                reset_user_output_memory(this, dep_memory_ptr(0));
            }
            return;
        } else if (_outputs[0] && dep_memory_ptr(0) &&
                   _network.get_engine().is_the_same_buffer(dep_memory(0), output_memory(0))) {
            if (mem_allocated()) {
                get_network().get_memory_pool().release_memory(_outputs[0].get(),
                        get_node().get_unique_id(), id(), get_network_id());
                _mem_allocated = false;
            }
            clear_output_memory();
            // Check users recursively and if the users is can_be_optimized && runtime_skippable
            // && output_memory of user is same as current input memory,
            // then reset the users output memory too.
            // Ex.
            // * iter0: node1(skipped)  -> node2(skipped) -> node3(skipped)
            // * iter1: node1(executed) -> node2(skipped) -> node3(executed)
            reset_user_output_memory(this, dep_memory_ptr(0));
        } else {
            // when this inst was not executed at the previous iteration,
            // Reset output memory because current output memory is invalid.
            if (prev_execution_skipped) {
                if (_outputs[0]) {
                    reset_user_output_memory(this, _outputs[0]);
                }
                clear_output_memory();
            }
        }
    }

    // update layout to ensure that it repsects paddings for correct allocation size
    if (_node_output_layout.data_padding.is_dynamic()) {
        auto update_padding = [](layout& orig_layout) {
            auto current_dims = orig_layout.get_padded_dims();

            std::vector<size_t> current_buf_shape;
            current_buf_shape.reserve(current_dims.size());
            std::transform(current_dims.begin(), current_dims.end(),
                        std::back_inserter(current_buf_shape), [](const tensor::value_type& el) { return static_cast<size_t>(el); });
            orig_layout = layout(ov::PartialShape(current_buf_shape), orig_layout.data_type, orig_layout.format);
        };

        update_padding(updated_layouts[0]);

        // Update scales and zero points buffers paddings, skipping beam_table
        if (_node->is_type<kv_cache>()) {
            for (size_t i = 2; i < updated_layouts.size(); ++i) {
                update_padding(updated_layouts[i]);
            }
        }
    }

    int32_t tmp_prealloc_count = get_prealloc_iter_num();
    // If we allocated too large memory, reclaim the memory.
    for (size_t i = 0; i < updated_layouts.size(); ++i) {
        bool reclaim = 0;
        size_t required_buffer_size = 0;
        if (_node->is_type<kv_cache>() && i != 1) {
            // Relax reclaiming condition for kv cache
            const auto& desc = _node->as<kv_cache>().get_primitive();
            auto prealloc_shape = updated_layouts[i].get_shape();
            const auto shape_rank = prealloc_shape.size();
            const auto seq_axis = i == 0 ? kv_cache_inst::get_sequence_axis(desc->concat_axis, shape_rank)
                                         : kv_cache_inst::get_scale_zp_sequence_axis();

            prealloc_shape[seq_axis] += tmp_prealloc_count;
            required_buffer_size = std::accumulate(prealloc_shape.begin(), prealloc_shape.end(), size_t(1), std::multiplies<size_t>());
        } else {
            required_buffer_size = (updated_layouts[i].get_linear_size());
        }
        if (required_buffer_size * 10 < _max_output_layout_count[i]) {
            reclaim = true;
        }
        if (reclaim) {
            GPU_DEBUG_TRACE_DETAIL << id() << ": Updated output[" << i << "] size " << updated_layouts[i].get_linear_size()
                                   << " is much smaller than current memory size! " << _max_output_layout_count[i]
                                   << "Reset memory of output " << i << std::endl;
            _max_output_layout_count[i] = 0;
        }
    }

    // Handle runtime dynamic concat optimization
    if (_node->is_type<concatenation>() && can_be_optimized() && allocation_done_by_other) {
        allocation_done_by_other = false;
        GPU_DEBUG_PROFILED_STAGE_MEMALLOC_INFO("concat_alloc_by_other");
        return;
    }

    for (size_t i = 0; i < actual_layouts.size(); ++i) {
        bool can_reuse_buffer = (_outputs[i] && updated_layouts[i].get_linear_size() <= _max_output_layout_count[i]);
        std::pair<bool, ov::Shape> prealloc_info;
        if (_node->is_type<kv_cache>() && i != 1) {
            const auto& desc = _node->as<kv_cache>().get_primitive();
            const auto shape_rank = updated_layouts[i].get_shape().size();
            const auto seq_axis = i == 0 ? kv_cache_inst::get_sequence_axis(desc->concat_axis, shape_rank)
                                         : kv_cache_inst::get_scale_zp_sequence_axis();

            prealloc_info = sp.predict_preallocation_shape(id(), updated_layouts[i], false, i, tmp_prealloc_count, seq_axis);
        } else {
            prealloc_info = sp.predict_preallocation_shape(id(), updated_layouts[i], can_reuse_buffer, i, tmp_prealloc_count);
        }
        if (prealloc_info.first && sp.can_preallocate(ov::shape_size(prealloc_info.second) * (dt_sizes_in_B[i]))) {
            updated_params.output_layouts[i] = updated_layouts[i].clone_with_other_shape(prealloc_info.second);
        }
        if (updated_params.output_layouts[i].get_linear_size() < updated_layouts[i].get_linear_size()) {
            updated_params.output_layouts[i] = updated_layouts[i];
        }

        if (can_reuse_buffer) {
            GPU_DEBUG_TRACE_DETAIL << id() << ": reuse previously allocated output buffer[" << i << "] - "
                                   << actual_layouts[i].get_linear_size() << "/" << _max_output_layout_count[i]
                                   << std::endl;
            if (_node->is_type<kv_cache>() && i != 1) {
                // kv_cache has already assigned memory.
                // No need to reinterpret output memory but need to update padding
                const auto& desc = _node->as<kv_cache>().get_primitive();
                auto& present_layout = _impl_params->output_layouts[i];
                const auto present_layout_rank = present_layout.get_partial_shape().size();
                const auto sequence_axis = i == 0 ? kv_cache_inst::get_sequence_axis(desc->concat_axis, present_layout_rank)
                                                  : kv_cache_inst::get_scale_zp_sequence_axis();

                auto max_pad = kv_cache_inst::get_max_pad(present_layout,
                                                          _max_output_layout_count[i],
                                                          sequence_axis,
                                                          i == 0 ? "present_layout" : "present_scales_layout");
                kv_cache_inst::update_pad(present_layout, max_pad, sequence_axis);
                GPU_DEBUG_TRACE_DETAIL << i << ". " << _impl_params->output_layouts[i].to_string() << std::endl;
                set_flag(ExecutionFlags::SHAPE_CHANGED);
            } else {
                _outputs[i] = _network.get_engine().reinterpret_buffer(*_outputs[i], actual_layouts[i]);
            }
            // TODO: check need_reset_output_memory per output
            if (need_reset_output_memory() && !can_be_optimized()) {
                GPU_DEBUG_TRACE_DETAIL << id() << " : Need reset output memory considering user" << std::endl;
                add_dep_event(_outputs[i]->fill(_network.get_stream()));
            }
            GPU_DEBUG_PROFILED_STAGE_MEMALLOC_INFO("reuse_buffer");
        } else {
            GPU_DEBUG_TRACE_DETAIL << id() << ": realloc output memory. " << std::endl;
            GPU_DEBUG_TRACE_DETAIL << " outputs[" << i << "] "
                                   << " Current buffer_size=" << _max_output_layout_count[i]
                                   << " Requested buffer_size=" << updated_layouts[i].get_linear_size()
                                   << std::endl;
            _outputs[i] = allocate_output(_network.get_engine(),
                                          _network.get_memory_pool(),
                                          *_node,
                                          updated_params,
                                          _runtime_memory_dependencies,
                                          get_network_id(),
                                          _network.is_internal(),
                                          i,
                                          need_reset_output_memory(),
                                          is_output_buffer(this, true),
                                          output_memory_ptr(i).get(),
                                          true);
            _max_output_layout_count[i] = updated_params.output_layouts[i].get_linear_size();
            set_flag(ExecutionFlags::MEMORY_CHANGED);
            GPU_DEBUG_CODE(std::string memalloc_info = "");
            GPU_DEBUG_CODE(memalloc_info += (((_outputs.size() > 1) ? ("o" + to_string(i) + ":") : "") +
                                  (_outputs[i]->from_memory_pool ? "from_pool" : "new_alloc"));)
            GPU_DEBUG_PROFILED_STAGE_MEMALLOC_INFO(memalloc_info);
        }
    }

    // Set variable memory same as output memory
    if (_node->is_type<kv_cache>()) {
        const auto& desc = _node->as<kv_cache>().get_primitive();
        auto& variable = get_network().get_variable(desc->variable_info.variable_id);
        auto present_layout = _impl_params->output_layouts[0];
        auto present_layout_rank = present_layout.get_partial_shape().size();
        const auto sequence_axis =
            kv_cache_inst::get_sequence_axis(desc->concat_axis, present_layout_rank);
        GPU_DEBUG_TRACE_DETAIL << id() << " is kv_cache => set the variable with newly allocated output memory"
                               << std::endl;
        bool axis_is_outer_most = true;
        for (auto dim = 0; dim < sequence_axis; ++dim) {
            if (present_layout.get_shape()[dim] > 1) {
                axis_is_outer_most = false;
                break;
            }
        }
        if (present_layout.data_padding._dynamic_dims_mask[sequence_axis] == 1) {
            // Apply padding of variable to make it be optimized in the next iteration
            auto max_pad = kv_cache_inst::get_max_pad(present_layout,
                                                      _max_output_layout_count[0],
                                                      sequence_axis,
                                                      "present_layout");
            if (max_pad > 0) {
                if (auto compressed_cache_variable = dynamic_cast<ov::intel_gpu::VariableStateIndirectKVCacheCompressed*>(&variable)) {
                    auto present_scales_layout = _impl_params->output_layouts[2];
                    const auto sequence_axis = kv_cache_inst::get_scale_zp_sequence_axis();

                    // In case of compressed KV-cache, calling update_impl for each iteration
                    // because of scales layout [batch, num_heads, seq_len, head_size], which requires proper
                    // dynamic padding updates
                    axis_is_outer_most = false;
                    kv_cache_inst::update_pad(present_scales_layout, max_pad, sequence_axis);

                    _impl_params->output_layouts[2] = present_scales_layout;
                    compressed_cache_variable->get_compression_scale_state()->set_memory(_outputs[2], present_scales_layout);
                    if (compressed_cache_variable->has_zp_state()) {
                        auto present_zp_layout = present_scales_layout;
                        present_zp_layout.data_type = _impl_params->output_layouts[3].data_type;

                        _impl_params->output_layouts[3] = present_zp_layout;
                        compressed_cache_variable->get_compression_zp_state()->set_memory(_outputs[3], present_zp_layout);
                    }
                }

                kv_cache_inst::update_pad(present_layout, max_pad, sequence_axis);
                if (!axis_is_outer_most) {
                    GPU_DEBUG_TRACE_DETAIL << id() << ": Update impl with new output padding" << std::endl;
                    set_flag(ExecutionFlags::SHAPE_CHANGED);
                    _impl_params->output_layouts[0] = present_layout;
                    update_impl(use_async_compilation());
                }
                GPU_DEBUG_TRACE_DETAIL << id() << ": Update variable " << variable.get_name()
                                       << "'s memory with allocated kv cache output: "
                                       << present_layout.to_short_string() << " is_set  = " << variable.is_set()
                                       << std::endl;
                variable.set_memory(_outputs[0], present_layout);
                _impl_params->_can_be_optimized = true;
                // No need to copy, still it can be optimized
                GPU_DEBUG_TRACE_DETAIL << id() << ": Set can_be_optimized = true " << std::endl;
            } else {
                GPU_DEBUG_TRACE_DETAIL << id() << ": Update variable " << variable.get_name()
                                       << "'s layout with allocated kv cache output: " << present_layout.to_short_string()
                                       << " (is_set  = " << variable.is_set() << ") " << std::endl;
                variable.set_memory(_outputs[0], present_layout);

                if (auto compressed_cache_variable = dynamic_cast<ov::intel_gpu::VariableStateIndirectKVCacheCompressed*>(&variable)) {
                    auto present_scales_layout = _impl_params->output_layouts[2];

                    compressed_cache_variable->get_compression_scale_state()->set_memory(_outputs[2], present_scales_layout);
                    if (compressed_cache_variable->has_zp_state()) {
                        auto present_zp_layout = present_scales_layout;
                        compressed_cache_variable->get_compression_zp_state()->set_memory(_outputs[3], present_zp_layout);
                    }
                }
            }
        } else {
            GPU_DEBUG_TRACE_DETAIL << id() << ": Update variable " << variable.get_name()
                                   << "'s layout with allocated kv cache output: " << present_layout.to_short_string()
                                   << " (is_set  = " << variable.is_set() << ") " << std::endl;
            variable.set_layout(present_layout);

            if (auto compressed_cache_variable = dynamic_cast<ov::intel_gpu::VariableStateIndirectKVCacheCompressed*>(&variable)) {
                auto present_scales_layout = _impl_params->output_layouts[2];

                compressed_cache_variable->get_compression_scale_state()->set_layout(present_scales_layout);
                if (compressed_cache_variable->has_zp_state()) {
                    auto present_zp_layout = present_scales_layout;
                    compressed_cache_variable->get_compression_zp_state()->set_layout(present_zp_layout);
                }
            }
        }
    }

    _mem_allocated = true;
    // intermediate memory allocation is required for primitives consisting of multiple kernels in dynamic case
    {
        if (_impl == nullptr)
            return;
        const auto& buffer_descs = _impl->get_internal_buffer_descs(*_impl_params);
        if (buffer_descs.empty())
            return;
        GPU_DEBUG_CODE(std::string memalloc_info = "");
        for (size_t i = 0; i < buffer_descs.size(); ++i) {
            auto need_lockable = buffer_descs[i].m_lockable;
            auto alloc_type = i < _intermediates_memory.size() ? _intermediates_memory[i]->get_allocation_type()
                                                               : allocation_type::unknown;
            bool can_reuse = true;
            can_reuse &= alloc_type != allocation_type::unknown &&
                         buffer_descs[i].m_layout.bytes_count() <= max_intermediates_memory_sizes[i];
            can_reuse &= (need_lockable && alloc_type != cldnn::allocation_type::usm_device) ||
                         (!need_lockable && alloc_type != cldnn::allocation_type::usm_host);

            if (can_reuse) {
                _intermediates_memory[i] = _network.get_engine().reinterpret_buffer(*_intermediates_memory[i], buffer_descs[i].m_layout);
               GPU_DEBUG_CODE(memalloc_info += ((_intermediates_memory.size() > 1) ? ("i" + to_string(i) + ":") : "") + "reuse_buffer");
            } else {
                // TODO: If there is a kernel which requires reset internal buffer in the future,
                // we'll need additional handle for that purpose like need_reset_output_memory
                const bool need_reset = false;
                if (i < _intermediates_memory.size()) {
                    _intermediates_memory[i] = allocate_internal_buffer(buffer_descs[i].m_layout, i, need_reset, need_lockable);
                    max_intermediates_memory_sizes[i] = _intermediates_memory[i]->size();
                } else {
                    // i-th layout has not been allocated yet
                    _intermediates_memory.push_back(allocate_internal_buffer(buffer_descs[i].m_layout, i, need_reset, need_lockable));
                    max_intermediates_memory_sizes.push_back(_intermediates_memory[i]->size());
                }
                GPU_DEBUG_CODE(memalloc_info +=
                               (((_intermediates_memory.size() > 1) ? ("i" + to_string(i) + ":") : "") +
                                (_intermediates_memory[i]->from_memory_pool ? "from_pool" : "new_alloc")));
            }
        }
        GPU_DEBUG_PROFILED_STAGE_MEMALLOC_INFO(memalloc_info);
    }
}

bool primitive_inst::use_async_compilation() {
    GPU_DEBUG_IF(get_config().get_disable_async_compilation()) {
        return false;
    }

    bool compile_fc_impls = _node->is_type<fully_connected>();
    if (compile_fc_impls) {
        const auto& fc_node = _node->as<fully_connected>();
        if (fc_node.get_primitive()->compressed_weights) {
            auto weights_dt = fc_node.weights().get_output_layout().data_type;
            auto input_shape = _impl_params->get_input_layout().get_shape();
            auto batch_size = std::accumulate(input_shape.begin(),
                                              input_shape.end() - 1,
                                              size_t{1},
                                              std::multiplies<size_t>());

            // Disable async compilation for all int4 FC, except in the case of batch_size == 1
            if (one_of(weights_dt, {data_types::i4, data_types::u4}) && batch_size != 1)
                compile_fc_impls = false;
        }
    }

    bool compile_gemm_impls = _node->is_type<gemm>();
    if (compile_gemm_impls) {
        // Do not async-compile if opt_gemm is chosen for iGPU
        // Do async-compile if it is to be executed from onednn
        compile_gemm_impls = _node->get_selected_impl() && _node->get_selected_impl()->get_kernel_name().find("gemm_ref") != std::string::npos;
        compile_gemm_impls |= _impls_factory->has(impl_types::onednn) && _node->get_selected_impl() && !_node->get_selected_impl()->is_onednn();
    }

    return (_node->is_type<convolution>() || compile_fc_impls || compile_gemm_impls ||
            (_node->is_type<softmax>() && _node->get_selected_impl() &&
             _node->get_selected_impl()->get_kernel_name().find("softmax_gpu_ref") != std::string::npos));
}

void primitive_inst::fill_shape_info_data(const layout& runtime_layout, const layout& node_layout, int32_t* shape_info_ptr, size_t& offset) {
    if (node_layout.is_static()) {
        GPU_DEBUG_TRACE_DETAIL << "tensor is static. Skipping" << std::endl;
        return;
    }
    const auto& pshape = runtime_layout.get_partial_shape();
    auto shape_info_fmt = format::get_default_format(layout::max_rank());
    auto shape_with_max_rank = layout::transform(pshape,
                                                 format::get_default_format(pshape.size()),
                                                 shape_info_fmt).to_shape();
    for (size_t j = 0; j < shape_with_max_rank.size(); ++j) {
        GPU_DEBUG_TRACE_DETAIL << " shape_info[" << offset << "] = " << shape_with_max_rank[j] << std::endl;
        shape_info_ptr[offset++] = static_cast<int32_t>(shape_with_max_rank[j]);
    }
    const auto& dynamic_pad = node_layout.data_padding._dynamic_dims_mask;
    const auto& data_padding = runtime_layout.data_padding;
    const auto& lower_pads = data_padding._lower_size;
    const auto& upper_pads = data_padding._upper_size;
    for (size_t j = 0; j < shape_with_max_rank.size(); ++j) {
        if (dynamic_pad[j] == 1) {
            GPU_DEBUG_TRACE_DETAIL << " shape_info[" << offset << "] = " << lower_pads[j]
                                   << "(pad_before for " << j << "-th dim)" << std::endl;
            shape_info_ptr[offset++] = lower_pads[j];  // pad_before
            GPU_DEBUG_TRACE_DETAIL << " shape_info[" << offset << "] = " << upper_pads[j]
                                   << "(pad_after for " << j << "-th dim)" << std::endl;
            shape_info_ptr[offset++] = upper_pads[j];  // pad_after
        }
    }
}

void primitive_inst::set_shape_info_memory_subbuffer(memory::ptr addr) {
    _shape_info_memory = addr;
}

void primitive_inst::allocate_shape_info_memory() {
    int64_t shape_elements = _node->get_total_shape_info_size();
    _shape_info_memory = _network.get_engine().allocate_memory(layout{{shape_elements}, data_types::i32, format::bfyx}, false);
}

void primitive_inst::update_shape_info_tensor(const kernel_impl_params& params) {
    if (!_shape_info_memory) {
        allocate_shape_info_memory();
    }

    mem_lock<int32_t> lock(_shape_info_memory, _network.get_stream());
    auto shape_info_ptr = lock.data();
    size_t offset = 0;
    for (size_t i = 0; i < _node->get_dependencies().size(); i++) {
        GPU_DEBUG_TRACE_DETAIL << id() << " : update shape_info for input[" << i << "]" << std::endl;
        const auto& node_in_lay = _node->get_input_layout(i);
        const auto& runtime_in_lay = params.input_layouts[i];
        fill_shape_info_data(runtime_in_lay, node_in_lay, shape_info_ptr, offset);
    }
    for (size_t i = 0; i < _node->get_output_layouts().size(); i++) {
        GPU_DEBUG_TRACE_DETAIL << id() << " : update shape_info for output[" << i << "]" << std::endl;
        const auto& node_out_lay = _node->get_output_layout(i);
        const auto& runtime_out_lay = params.output_layouts[i];
        fill_shape_info_data(runtime_out_lay, node_out_lay, shape_info_ptr, offset);
    }
}

void primitive_inst::update_impl(bool use_async_compilation) {
    OV_ITT_SCOPED_TASK(ov::intel_gpu::itt::domains::intel_gpu_plugin, openvino::itt::handle("update_impl: " + id()));
    GPU_DEBUG_PROFILED_STAGE(instrumentation::pipeline_stage::update_implementation);
    auto prev_impl_str =  _impl != nullptr ? _impl->get_kernel_name() : "nullptr";

    // no need to update impl for optimized out primitive
    if (_impl != nullptr && can_be_optimized()) {
        GPU_DEBUG_TRACE_DETAIL << id() << " Skip impl update: primitive is optimized out" << std::endl;
        set_flag(ExecutionFlags::IMPL_CHANGED, get_flag(ExecutionFlags::SHAPE_CHANGED));
        return;
    }

    // Assume that we have already picked optimal impl
    if (!get_flag(ExecutionFlags::SHAPE_CHANGED) && _impl && _impl->is_dynamic() && !use_async_compilation) {
        GPU_DEBUG_TRACE_DETAIL << id() << " Skip impl update: shape not changed, optimal static impl is used" << std::endl;
        unset_flag(ExecutionFlags::IMPL_CHANGED);
        return;
    }

    if (!_node->is_type<data>() && !(_node->is_type<mutable_data>() && _node->get_dependencies().empty())) {
#ifdef ENABLE_ONEDNN_FOR_GPU
        if (_impls_factory->has(impl_types::onednn)) {
            auto attrs_onednn = std::make_shared<dnnl::primitive_attr>();
            std::vector<cldnn::fused_primitive_desc_onednn> fused_desc_onednn;
            get_node().create_onednn_primitive_attributes(_impl_params->fused_desc,
                                                            attrs_onednn,
                                                            fused_desc_onednn,
                                                            _impl_params.get());
            _impl_params->attrs_onednn = attrs_onednn;
            {
                auto& fused_prims_onednn = _impl_params->fused_desc_onednn;
                fused_prims_onednn.erase(fused_prims_onednn.begin(), fused_prims_onednn.end());
                fused_prims_onednn.insert(fused_prims_onednn.end(), fused_desc_onednn.begin(), fused_desc_onednn.end());
            }
        }
#endif

        _impl = _impls_factory->get_primitive_impl_for_params(*this, *_impl_params, use_async_compilation);
        GPU_DEBUG_TRACE_DETAIL << id() << " impl update: was: " << prev_impl_str << " now: " << _impl->get_kernel_name() << std::endl;
    }

    set_flag(ExecutionFlags::IMPL_CHANGED);
}

void primitive_inst::update_paddings() {
    auto reset_pad = [](kernel_impl_params& params, const program_node* node, size_t idx = 0) {
        params.output_layouts[idx].data_padding = node->get_output_layout(idx).data_padding;
    };
    if (_node->is_type<read_value>() || _node->is_type<kv_cache>()) {
        auto variable_id = _node->is_type<read_value>() ? (_node->as<read_value>().get_primitive()->variable_id)
                                                        : (_node->as<kv_cache>().get_primitive()->variable_info.variable_id);
        auto& variable = get_network().get_variable(variable_id);
        // Reset paddings for read_value and users with dynamic pad when variable is reset
        // to avoid wrong pad used for some nodes due to pad propagation logic (which uses previous iter pad values)
        if (!variable.is_set()) {
            primitive_inst* inst = this;
            while (inst) {
                reset_pad(*inst->_impl_params, inst->_node);
                if (inst == this) {
                    if (auto compressed_cache_variable = dynamic_cast<ov::intel_gpu::VariableStateIndirectKVCacheCompressed*>(&variable)) {
                        const size_t scale_idx = _node->is_type<read_value>() ? 1 : 2;
                        reset_pad(*inst->_impl_params, inst->_node, scale_idx);
                        if (compressed_cache_variable->has_zp_state()) {
                            reset_pad(*inst->_impl_params, inst->_node, scale_idx + 1);
                        }
                    }
                }
                auto& users = inst->_node->get_users();
                if (users.size() == 1 && users.front()->get_output_layout(0).data_padding.is_dynamic()) {
                    inst = inst->get_user_insts().front();
                } else {
                    inst = nullptr;
                }
            }
        }
        return;
    }

    if (_node->is_type<gather>() && _impl_params->output_layouts[0].data_padding.is_dynamic()) {
        if (can_be_optimized())
            _impl_params->output_layouts[0] = _impl_params->input_layouts[0];
        else
            reset_pad(*_impl_params, _node);
        return;
    }
    // Reset paddings used in the previous iteration for crop before executing do_runtime_in_place_crop
    for (auto u : get_user_insts()) {
        if (u->get_node().is_type<crop>() && u->_impl_params->output_layouts[0].data_padding.is_dynamic()) {
            if (u->get_node().can_be_optimized()) {
                reset_pad(*u->_impl_params, u->_node);
            }
        }
    }
}

void primitive_inst::do_runtime_skip_reorder() {
    OV_ITT_SCOPED_TASK(ov::intel_gpu::itt::domains::intel_gpu_plugin, openvino::itt::handle("do_runtime_skip_reorder: " + id()));
    GPU_DEBUG_IF(get_config().get_disable_runtime_skip_reorder()) {
        return;
    }
    if (can_be_optimized())
        return;

    if (_impl_params->fused_desc.size() > 0)
        return;

    // set successive reorder can_be_optimized if layouts are same
    for (auto u : get_user_insts()) {
        if (u->get_node().is_type<reorder>()) {
            if (u->get_node().can_be_optimized() && u->get_node().is_runtime_skippable()) {
                auto out_port_idx = u->get_node().get_dependency_with_port(0).second;
                // If current node's output_node is not dynamic, the memory is already allocated at build time
                auto alloc_type = allocation_type::unknown;
                if (!get_node().is_dynamic_output_layout(out_port_idx) && static_cast<int64_t>(_outputs.size()) > out_port_idx) {
                    alloc_type = _outputs[out_port_idx]->get_allocation_type();
                }
                if (alloc_type == allocation_type::usm_device && u->is_output()) {
                    u->set_can_be_optimized(false);
                    GPU_DEBUG_TRACE_DETAIL << "[do runtime skip reorder] user " << u->id()
                                                << " cannot be optimized for that " << u->id() << " is reorder and output node" << std::endl;
                    continue;
                }
                GPU_DEBUG_TRACE_DETAIL << "[do runtime skip reorder] update shape for user " << u->id() << std::endl;
                u->update_shape();
                u->update_shape_done_by_other = true;

                if (u->_impl_params->get_input_layout() == u->_impl_params->get_output_layout()) {
                    std::function<void(std::vector<primitive_inst*>)> update_memory_dependencies;
                    update_memory_dependencies = [&](std::vector<primitive_inst*> users) {
                        for (auto& user : users) {
                            GPU_DEBUG_TRACE_DETAIL << "[do runtime skip reorder] add " << id() << " to restriction list of " << user->id() << std::endl;
                            user->_runtime_memory_dependencies.insert(get_node().get_unique_id());
                            if (user->can_be_optimized())
                                update_memory_dependencies(user->get_user_insts());
                        }
                    };

                    update_memory_dependencies(u->get_user_insts());
                    u->set_can_be_optimized(true);
                    GPU_DEBUG_TRACE_DETAIL << "[do runtime skip reorder] set user " << u->id() << " as can_be_optimized" << std::endl;
                } else {
                    u->set_can_be_optimized(false);
                    GPU_DEBUG_TRACE_DETAIL << "[do runtime skip reorder] user " << u->id()
                                                << " cannot be optimized for the mismatch between input layout and output layout" << std::endl;
                    GPU_DEBUG_TRACE_DETAIL << "[do runtime skip reorder]  * input_layout  : "
                                                << u->_impl_params->get_input_layout().to_short_string() << std::endl;
                    GPU_DEBUG_TRACE_DETAIL << "[do runtime skip reorder]  * output_layout : "
                                                << u->_impl_params->get_output_layout().to_short_string() << std::endl;
                }
            }
        }
    }
}

void primitive_inst::do_runtime_in_place_kv_cache() {
    OV_ITT_SCOPED_TASK(ov::intel_gpu::itt::domains::intel_gpu_plugin, openvino::itt::handle("do_runtime_in_place_kv_cache: " + id()));
    if (!_node->is_type<kv_cache>())
        return;

    _impl_params->_can_be_optimized = false;

    if (_impl_params->get_input_layout(0).count() == 0) {
        return;
    }
    const auto& desc = _node->as<kv_cache>().get_primitive();
    auto& past_layout = _impl_params->input_layouts[0];
    auto& new_layout = _impl_params->input_layouts[1];
    auto& present_layout = _impl_params->output_layouts[0];
    const auto& gather_axis = desc->gather_axis;
    const auto prev_batch_size = static_cast<size_t>(past_layout.get_shape()[gather_axis]);
    const auto beam_size = static_cast<size_t>(present_layout.get_shape()[gather_axis]);
    if (prev_batch_size != beam_size) {
        // If the previous batch size is not same as beam size, need explicit concat
        _impl_params->_can_be_optimized = false;
        return;
    }

    auto sequence_axis = kv_cache_inst::get_sequence_axis(desc->concat_axis, past_layout.get_partial_shape().size());
    if (present_layout.data_padding._dynamic_dims_mask[sequence_axis] != 1)
        return;

    GPU_DEBUG_TRACE_DETAIL << "[do runtime kv_cache opt] " << id() << " initial present_layout : " << present_layout.to_string() << std::endl;
    GPU_DEBUG_TRACE_DETAIL << "[do runtime kv_cache opt] " << id() << " initial past_layout : " << past_layout.to_string() << std::endl;
    auto max_pad = kv_cache_inst::get_max_pad(past_layout, _deps[0].first->_max_output_layout_count[0], sequence_axis, "past_layout");
    const auto new_seq_len = static_cast<int64_t>(new_layout.get_shape()[sequence_axis]);
    // In chatbot scenario, when chat history must be stored in kvcache, new_seq_len may not be 1 even if max_pad is greater than 0
    if (max_pad - new_seq_len >= 0) {
        kv_cache_inst::update_pad(present_layout, max_pad - new_seq_len, sequence_axis);
        GPU_DEBUG_TRACE_DETAIL << "[do runtime_in_place_kv_cache] " << id() << " Updated present_layout's pad : " << present_layout.to_string() << std::endl;
        auto& variable = get_network().get_variable(desc->variable_info.variable_id);
        variable.set_layout(present_layout);

        if (desc->compressed) {
            auto compressed_cache_variable = dynamic_cast<ov::intel_gpu::VariableStateIndirectKVCacheCompressed*>(&variable);
            auto& present_scales_layout = _impl_params->output_layouts[2];
            const auto sequence_axis = kv_cache_inst::get_scale_zp_sequence_axis();
            kv_cache_inst::update_pad(present_scales_layout, max_pad - new_seq_len, sequence_axis);
            GPU_DEBUG_TRACE_DETAIL << "[do runtime_in_place_kv_cache] " << id()
                                   << " Updated present_scale_layout's pad : " << present_scales_layout.to_string() << std::endl;

            compressed_cache_variable->get_compression_scale_state()->set_layout(present_scales_layout);
            if (desc->get_compression_zp_inputs_num() > 0) {
                auto& present_zp_layout = _impl_params->output_layouts[3];
                kv_cache_inst::update_pad(present_zp_layout, max_pad - new_seq_len, sequence_axis);
                GPU_DEBUG_TRACE_DETAIL << "[do runtime_in_place_kv_cache] " << id()
                                       << " Updated present_zp_layout's pad : " << present_scales_layout.to_string() << std::endl;

                compressed_cache_variable->get_compression_zp_state()->set_layout(present_zp_layout);
            }
        }

        GPU_DEBUG_TRACE_DETAIL << "[do_runtime_in_place_kv_cache] " << id() << "Updated variable with present_layout"
                               << variable.get_layout().to_string() << " is_set  = " << variable.is_set() << std::endl;
        if (past_layout.data_padding._upper_size[sequence_axis] > 0 && variable.is_set()) {
            kv_cache_inst::update_pad(past_layout, max_pad, sequence_axis);
            _impl_params->_can_be_optimized = true;

            if (desc->compressed) {
                auto& past_scale_layout = _impl_params->input_layouts[3];
                const auto sequence_axis = kv_cache_inst::get_scale_zp_sequence_axis();
                kv_cache_inst::update_pad(past_scale_layout, max_pad, sequence_axis);

                if (desc->get_compression_zp_inputs_num() > 0) {
                    auto& past_zp_layout = _impl_params->input_layouts[4];
                    kv_cache_inst::update_pad(past_zp_layout, max_pad, sequence_axis);
                }
            }
            GPU_DEBUG_TRACE_DETAIL << "[do_runtime_in_place_kv_cache] " << id() << " Updated past layout's pad : " << past_layout.to_string() << std::endl;
        }
    }
    GPU_DEBUG_TRACE_DETAIL << "[do runtime kv_cache opt] " << id() << " can be optimized: " << _impl_params->_can_be_optimized << std::endl;
}

void primitive_inst::do_runtime_skip_gather() {
    // Check pattern
    if (!get_node().is_type<gather>()
        || !get_node().is_runtime_skippable()
        || _impl_params->has_fused_primitives()
        || _impl_params->get_input_layout(0).data_type != _impl_params->get_output_layout().data_type
        || get_node().get_dependency(1).is_constant() || get_node().get_dependency(1).is_type<data>())
        return;

    GPU_DEBUG_TRACE_DETAIL << "[do_runtime_skip_gather] " << id() << " : check optimizability" << std::endl;

    auto input_shape = _impl_params->get_input_layout(0).get_shape();
    auto axis = _impl_params->typed_desc<gather>()->axis;
    auto idx_id = get_node().get_dependency(1).id();
    auto idx_shape = _impl_params->get_input_layout(1).get_shape();
    auto idx_rank = idx_shape.size();

    if (_impl_params->get_input_layout(0).count() == 0) {
        GPU_DEBUG_TRACE_DETAIL << "-- Cannot optimize because input is empty " << _impl_params->get_input_layout(0).to_short_string() << std::endl;
        set_can_be_optimized(false);
        return;
    }

    if (idx_rank != 1) {
        GPU_DEBUG_TRACE_DETAIL << "-- Cannot optimize because of its indices rank " << idx_rank << std::endl;
        set_can_be_optimized(false);
        return;
    }

    // Check runtime shape (need to reset can_be_optimized)
    if (idx_shape[0] != input_shape[axis]) {
        set_can_be_optimized(false);
        GPU_DEBUG_TRACE_DETAIL << "--- Cannot optimize because input shape[0] " << idx_shape[0] << " != input_shape[axis]" << input_shape[axis] << std::endl;
        return;
    }

    // If the overhead for checking the index is bigger than doing gather itself, it does not make sense for skipping
    const int MAX_INDICES_SIZE = 10*1024;
    if (input_shape[axis] > MAX_INDICES_SIZE) {
        GPU_DEBUG_TRACE_DETAIL << "--- Cannot optimize because data length along with the axis is too big" << input_shape[axis] << std::endl;
        set_can_be_optimized(false);
        return;
    }
    if (input_shape[axis] != 1) {
        auto queue_type = get_network().get_stream().get_queue_type();
        if (queue_type == QueueTypes::out_of_order)
            get_network().get_stream().wait_for_events({_deps[1].first->get_impl_params()->out_event});
        else
            _network.get_stream().finish();
        mem_lock<int32_t, mem_lock_type::read> idx_data(dep_memory_ptr(1), _network.get_stream());
        for (int64_t i = 0; i < static_cast<int32_t>(idx_shape[0]); ++i) {
            if (idx_data[i] != i) {
                GPU_DEBUG_TRACE_DETAIL << "--- Cannot optimize because idx_data [" << i << "] (" << idx_data[i] << ") != " << i << std::endl;
                if (_impl_params->output_layouts[0].data_padding.is_dynamic())
                    _impl_params->output_layouts[0].data_padding = padding();
                // for runtime skippable nodes, if previous iter is skipped while this iter not, its output memory needs to be revalidate
                // as memory opt/release may be applied for these nodes to reduce memory footprint in previous iters
                if (can_be_optimized()) {
                    set_flag(ExecutionFlags::SHAPE_CHANGED);
                }
                set_can_be_optimized(false);
                return;
            }
        }
    }
    // propagate input layout including correct paddings.
    _impl_params->output_layouts[0] = _impl_params->input_layouts[0];
    GPU_DEBUG_TRACE_DETAIL << "[do_runtime_skip_gather] " << id() << " : can_be_optimized" << std::endl;
    GPU_DEBUG_TRACE_DETAIL << "            - Input layout : " << _impl_params->get_input_layout(0).to_short_string() << std::endl;
    GPU_DEBUG_TRACE_DETAIL << "            - Indices layout : " << _impl_params->get_input_layout(1).to_short_string() << std::endl;
    GPU_DEBUG_TRACE_DETAIL << "            - Gather axis : " << axis << std::endl;
    GPU_DEBUG_TRACE_DETAIL << "            - Output layout : " << _impl_params->get_output_layout().to_short_string() << std::endl;
    set_can_be_optimized(true);
}

void primitive_inst::do_runtime_skip_permute() {
    OV_ITT_SCOPED_TASK(ov::intel_gpu::itt::domains::intel_gpu_plugin, openvino::itt::handle("do_runtime_skip_permute: " + id()));
    // Check pattern
    if (!get_node().is_type<permute>()
        || is_output()
        || !get_node().is_runtime_skippable()
        || _impl_params->has_fused_primitives()
        || _impl_params->get_input_layout(0).data_type != _impl_params->get_output_layout().data_type)
        return;

    GPU_DEBUG_TRACE_DETAIL << "[do_runtime_skip_permute] " << id() << " : check optimizability" << std::endl;
    auto desc = _node->as<permute>().get_primitive();
    auto input_shape = _impl_params->get_input_layout(0).get_shape();
    const auto& permute_order = desc->permute_order;
    // Can skip if the transposed dim keeps the original order
    bool can_skip = true;
    auto permute_dest = permute_order;
    for (size_t i = 0; i < permute_order.size(); ++i) {
        permute_dest[permute_order[i]] = static_cast<uint16_t>(i);
    }
    int16_t prev_dim = -1;
    for (size_t i = 0; i < permute_dest.size(); ++i) {
        if (input_shape[i] > 1) {
            if (permute_dest[i] < prev_dim) {
                can_skip = false;
                break;
            }
            prev_dim = permute_dest[i];
        }
    }
    GPU_DEBUG_TRACE_DETAIL << "[do_runtime_skip_permute] " << id() << " : can_be_optimized ? " << can_skip << std::endl;
    GPU_DEBUG_TRACE_DETAIL << "            - Input layout : " << _impl_params->get_input_layout(0).to_short_string() << std::endl;
    GPU_DEBUG_TRACE_DETAIL << "            - Output layout : " << _impl_params->get_output_layout().to_short_string() << std::endl;
    GPU_DEBUG_TRACE_DETAIL << "            - permute order : ";
    for (auto order : permute_order) {
        GPU_DEBUG_TRACE_DETAIL << order << ",";
    }
    GPU_DEBUG_TRACE_DETAIL << std::endl;
    set_can_be_optimized(can_skip);
}

void primitive_inst::do_runtime_skip_strided_slice() {
    OV_ITT_SCOPED_TASK(ov::intel_gpu::itt::domains::intel_gpu_plugin, openvino::itt::handle("do_runtime_skip_strided_slice: " + id()));
    // Check pattern
    if (!get_node().is_type<strided_slice>() || !get_node().is_runtime_skippable())
        return;

    GPU_DEBUG_TRACE_DETAIL << "[do_runtime_skip_strided_slice] " << id() << " : check optimizability" << std::endl;
    auto input_layout = _impl_params->get_input_layout(0);
    auto output_layout = _impl_params->get_output_layout();

    // Check runtime shape (need to reset can_be_optimized)
    if (input_layout != output_layout) {
        set_can_be_optimized(false);
        GPU_DEBUG_TRACE_DETAIL << "--- Cannot optimize because input layout(" << input_layout.to_short_string()
                               << ") != output layout(" << output_layout.to_short_string() << ")" << std::endl;
        return;
    }

    GPU_DEBUG_TRACE_DETAIL << "[do_runtime_skip_strided_slice] " << id() << " : can_be_optimized" << std::endl;
    GPU_DEBUG_TRACE_DETAIL << "            - Input layout : " << _impl_params->get_input_layout(0).to_short_string() << std::endl;
    GPU_DEBUG_TRACE_DETAIL << "            - Output layout : " << _impl_params->get_output_layout().to_short_string() << std::endl;
    set_can_be_optimized(true);
}

void primitive_inst::do_runtime_skip_broadcast() {
    OV_ITT_SCOPED_TASK(ov::intel_gpu::itt::domains::intel_gpu_plugin, openvino::itt::handle("do_runtime_skip_broadcast: " + id()));
    // Check pattern
    if (!get_node().is_type<broadcast>() || !get_node().is_runtime_skippable())
        return;

    GPU_DEBUG_TRACE_DETAIL << "[do_runtime_skip_broadcast] " << id() << " : check optimizability" << std::endl;
    auto input_layout = _impl_params->get_input_layout(0);
    auto output_layout = _impl_params->get_output_layout();

    // Check runtime shape (need to reset can_be_optimized)
    if (input_layout != output_layout) {
        set_can_be_optimized(false);
        GPU_DEBUG_TRACE_DETAIL << "--- Cannot optimize because input layout(" << input_layout.to_short_string()
                               << ") != output layout(" << output_layout.to_short_string() << ")" << std::endl;
        return;
    }

    GPU_DEBUG_TRACE_DETAIL << "[do_runtime_skip_broadcast] " << id() << " : can_be_optimized" << std::endl;
    GPU_DEBUG_TRACE_DETAIL << "            - Input layout : " << _impl_params->get_input_layout(0).to_short_string() << std::endl;
    GPU_DEBUG_TRACE_DETAIL << "            - Output layout : " << _impl_params->get_output_layout().to_short_string() << std::endl;
    set_can_be_optimized(true);
}

void primitive_inst::do_runtime_in_place_concat() {
     auto has_subgraph_dependency = [](std::vector<std::pair<const cldnn::primitive_inst*, int>> dependencies) {
        for (auto dependency : dependencies) {
            if (dependency.first && dependency.first->get_node().is_in_shape_of_subgraph()) {
                return true;
            }
        }
        return false;
    };
    OV_ITT_SCOPED_TASK(ov::intel_gpu::itt::domains::intel_gpu_plugin, openvino::itt::handle("do_runtime_in_place_concat: " + id()));
    GPU_DEBUG_IF(get_config().get_disable_runtime_buffer_fusing()) {
        return;
    }
    if (update_shape_done_by_other) {
        return;
    }
    if (get_users().size() != 1) return;

    auto concat_inst = get_user_insts().front();

    if (!concat_inst->get_node().is_type<concatenation>() || !(concat_inst->get_node().can_be_optimized() && concat_inst->get_node().is_runtime_skippable()))
        return;

    if (has_subgraph_dependency(concat_inst->dependencies())) {
        concat_inst->set_can_be_optimized(false);
        return;
    }
    // Currently does not support cascaded concats
    std::vector<primitive_inst*> concat_preds;
    for (auto pred : concat_inst->_deps) {
        concat_preds.push_back(pred.first);
    }

    GPU_DEBUG_TRACE_DETAIL << "[In place concat] Preparing for runtime buffer fusing" << std::endl;
    // Do shape_infer for all concat's preds and concat
    for (auto pred : concat_preds) {
        if (!pred->update_shape_done_by_other) {
            GPU_DEBUG_TRACE_DETAIL << "[In place concat] update shape for " << pred->id() << std::endl;
            pred->update_shape();
            pred->update_shape_done_by_other = true;
        }
    }
    GPU_DEBUG_TRACE_DETAIL << "[In place concat] update shape for " << concat_inst->id() << std::endl;
    concat_inst->update_shape();
    concat_inst->update_shape_done_by_other = true;
    layout concat_layout = concat_inst->_impl_params->get_output_layout();

    std::vector<kernel_impl_params> pred_params;
    std::vector<layout> preds_layouts;
    for (const auto& pred : concat_inst->_deps) {
        pred_params.push_back(*pred.first->_impl_params);
        preds_layouts.push_back(pred.first->_impl_params->get_output_layout());
    }

    if (!concat_inst->get_flag(ExecutionFlags::SHAPE_CHANGED))
        return;

    if (!concat_in_place_optimization::match(concat_inst->get_node(), *concat_inst->_impl_params, pred_params, true)) {
        concat_inst->set_can_be_optimized(false);
        GPU_DEBUG_TRACE_DETAIL << "[In place concat] " << concat_inst->id() << " cannot be optimized " << std::endl;
        return;
    }

    auto concat_axis = concat_inst->_impl_params->typed_desc<concatenation>()->axis;
    concat_in_place_optimization::update_in_place_concat_paddings(concat_layout, preds_layouts, concat_axis, true);
    size_t i = 0;
    for (auto& dep : concat_inst->_deps) {
        if (_impl_params->output_layouts[0] != preds_layouts[i]) {
            dep.first->set_flag(ExecutionFlags::SHAPE_CHANGED);
            dep.first->_impl_params->output_layouts[0] = preds_layouts[i];
        }
        GPU_DEBUG_TRACE_DETAIL << "[In place concat] Update padding of pred " << i << " : "
                               << dep.first->_impl_params->output_layouts[0].to_string() << std::endl;
        ++i;
    }
    concat_inst->_impl_params->output_layouts[0] = concat_layout; // TODO : Once this primitive_inst::can_be_optimized, consolidate it to impl_params->optimized

    concat_inst->set_can_be_optimized(true);
    GPU_DEBUG_TRACE_DETAIL << "[In place concat] " << concat_inst->id() << ": can_be_optimized " << std::endl;
}

void primitive_inst::do_runtime_skip_scatter_update() {
    OV_ITT_SCOPED_TASK(ov::intel_gpu::itt::domains::intel_gpu_plugin, openvino::itt::handle("do_runtime_skip_scatter_update: " + id()));
    // Check pattern
    if (!(get_node().is_type<scatter_update>()
        || get_node().is_type<scatter_elements_update>()
        || get_node().is_type<scatter_nd_update>())
        || !get_node().is_runtime_skippable())
        return;

    GPU_DEBUG_TRACE_DETAIL << "[do_runtime_skip_scatter_update] " << id() << " : check optimizability" << std::endl;
    const auto& input_layout = _impl_params->get_input_layout(0);
    const auto& output_layout = _impl_params->get_output_layout(0);
    const auto& idx_layout = _impl_params->get_input_layout(1);
    const auto& update_layout = _impl_params->get_input_layout(2);

    if ((idx_layout.count() > 0 && update_layout.count() > 0) || (get_node().is_type<scatter_elements_update>() && input_layout != output_layout)) {
        // set shape_change to realloc memory for same input shapes
        if (can_be_optimized()) {
            set_flag(ExecutionFlags::SHAPE_CHANGED);
        }
        set_can_be_optimized(false);
        GPU_DEBUG_TRACE_DETAIL << "--- Cannot optimize because idx_layout (" << idx_layout.to_short_string()
                        << ") and update_layout(" << update_layout.to_short_string() << ") are not zero"
                        "or input layout is different than output layout" << std::endl;
        return;
    }

    GPU_DEBUG_TRACE_DETAIL << "[do_runtime_skip_scatter_update] " << id() << " : can_be_optimized" << std::endl;
    GPU_DEBUG_TRACE_DETAIL << "            - Input layout  : " << _impl_params->get_input_layout(0).to_short_string() << std::endl;
    GPU_DEBUG_TRACE_DETAIL << "            - Idx layout    : " << _impl_params->get_input_layout(1).to_short_string() << std::endl;
    GPU_DEBUG_TRACE_DETAIL << "            - Update layout : " << _impl_params->get_input_layout(2).to_short_string() << std::endl;
    GPU_DEBUG_TRACE_DETAIL << "            - Output layout : " << _impl_params->get_output_layout().to_short_string() << std::endl;
    // set shape_change to realloc memory for same input shapes
    if (!can_be_optimized()) {
        set_flag(ExecutionFlags::SHAPE_CHANGED);
    }
    set_can_be_optimized(true);
}

void primitive_inst::do_runtime_in_place_crop() {
    OV_ITT_SCOPED_TASK(ov::intel_gpu::itt::domains::intel_gpu_plugin, openvino::itt::handle("do_runtime_in_place_crop: " + id()));
    GPU_DEBUG_IF(get_config().get_disable_runtime_buffer_fusing()) {
        return;
    }

    for (auto u : get_user_insts()) {
        if (u->get_node().is_type<crop>()) {
            if (u->get_node().can_be_optimized()) {
                GPU_DEBUG_TRACE_DETAIL << "[In place crop] update shape for " << u->id() << std::endl;
                u->update_shape();
                u->update_shape_done_by_other = true;

                const auto& crop_users = u->get_user_insts();
                std::pair<const program_node*, layout> user_info;
                if (crop_users.front()->get_node().is_type<reshape>()) {
                    OPENVINO_ASSERT(crop_users.size() == 1, "[GPU] Expected number of reshape users is 1, but it is ", crop_users.size());
                    auto reshape_inst = crop_users.front();
                    if (!reshape_inst->update_shape_done_by_other) {
                        GPU_DEBUG_TRACE_DETAIL << "[In place crop] update shape for " << reshape_inst->id() << std::endl;
                        reshape_inst->update_shape();
                        reshape_inst->update_shape_done_by_other = true;
                        user_info.first = &reshape_inst->get_node();
                        user_info.second = reshape_inst->_impl_params->get_output_layout();
                    }
                }

                layout crop_layout = u->_impl_params->get_output_layout();
                auto pred_layout = _impl_params->get_output_layout();
                if (!crop_in_place_optimization::match(u->get_node(), *u->_impl_params, pred_layout, true)) {
                    u->set_can_be_optimized(false);
                    GPU_DEBUG_TRACE_DETAIL << "[In place crop] " << u->id() << " cannot be optimized " << std::endl;
                    continue;
                }

                auto crop_axis = u->_impl_params->typed_desc<crop>()->axis;
                auto offsets = u->_impl_params->input_offsets[0];
                if (crop_in_place_optimization::can_crop_be_optimized_along_feature(crop_layout, pred_layout)) {
                    // TODO: If crop is optimized out w/ data padding along feature and crop's user is reshape
                    // manual dynamic padding update to reshape output layout is not currently supported
                    if (user_info.first) {
                        u->set_can_be_optimized(false);
                        GPU_DEBUG_TRACE_DETAIL << "[In place crop] " << u->id() << " cannot be optimized " << std::endl;
                        continue;
                    }
                    crop_in_place_optimization::update_in_place_crop_padding_along_feature(u->get_node(), crop_layout, pred_layout, offsets, crop_axis, true);
                } else if (crop_in_place_optimization::can_crop_be_optimized_simple_data_format(crop_layout, pred_layout)) {
                    crop_in_place_optimization::update_in_place_crop_padding_simple_data_format(crop_layout, pred_layout, user_info, offsets, crop_axis, true);
                    if (user_info.first) {
                        auto reshape_inst = crop_users.front();
                        reshape_inst->_impl_params->output_layouts[0] = user_info.second;
                        reshape_inst->set_flag(ExecutionFlags::SHAPE_CHANGED);
                    }
                } else {
                    u->set_can_be_optimized(false);
                    GPU_DEBUG_TRACE_DETAIL << "[In place crop] " << u->id() << " cannot be optimized " << std::endl;
                    continue;
                }
                u->_impl_params->output_layouts[0] = crop_layout;
                u->set_can_be_optimized(true);
                GPU_DEBUG_TRACE_DETAIL << "[In place crop] " << u->id() << ": can_be_optimized " << std::endl;
            }
        }
    }
}

bool primitive_inst::has_inner_networks() const {
    return (_impl_params->inner_nets.size() > 0);
}

void primitive_inst::add_dep_events(const std::vector<event::ptr>& events) {
    for (auto ev : events)
        add_dep_event(std::move(ev));
}

void primitive_inst::add_dep_event(event::ptr ev) {
    if (ev)
        _impl_params->dep_events.push_back(ev);
}

void primitive_inst::set_out_event(event::ptr&& ev) {
    _impl_params->out_event = ev;
}

void primitive_inst::reset_events() {
    _impl_params->dep_events.clear();
    _impl_params->out_event = nullptr;
}

void primitive_inst::set_flag(size_t flag, bool value) {
    _impl_params->flags.set(flag, value);
}
void primitive_inst::unset_flag(size_t flag) {
    _impl_params->flags.set(flag, false);
}

bool primitive_inst::get_flag(size_t flag) const {
    return _impl_params->flags.test(flag);
}

void primitive_inst::reset_flags() {
    _impl_params->flags.reset();
}

void primitive_inst::prepare_primitive() {
    OV_ITT_SCOPED_TASK(ov::intel_gpu::itt::domains::intel_gpu_plugin, openvino::itt::handle("primitive_inst::execute: " + id()));
    const auto& primitive_id = id();
    OPENVINO_ASSERT(_has_valid_input, primitive_id, " has invalid/unset input");
    GPU_DEBUG_TRACE_DETAIL << "-----------------------------------------------------------------" << std::endl;
    GPU_DEBUG_TRACE_DETAIL << "Execute " << id() << " (type: " << _impl_params->desc->type_string() << ") " << std::endl;
    for (size_t i = 0; i < _deps.size(); ++i) {
        GPU_DEBUG_TRACE_DETAIL << "- inputs[" << i << "] : " <<  _deps[i].first->id() << std::endl;
    }
    GPU_DEBUG_TRACE_DETAIL << "-----------------------------------------------------------------" << std::endl;

    // If it is optimized out or skipped for zero dimension at the previous iteration,
    // Set this flag true to reset output memory in realloc_if_needed.
    const bool prev_execution_skipped = can_be_optimized()
                        || (_impl_params->output_layouts[0].is_static() && _impl_params->output_layouts[0].count() == 0);
    const auto orig_outputs = _outputs;
    if ((is_dynamic() || _node->is_in_shape_of_subgraph()) && !has_inner_networks()) {
        do_runtime_in_place_concat();
        OPENVINO_ASSERT(_node != nullptr, "[GPU] Invalid primitive_inst object for dynamic shapes case: program_node can't be null");
        update_shape();

        if (_impl_params->output_layouts[0].count() == 0) {
            GPU_DEBUG_TRACE_DETAIL << id() << " : Skipping because output data is empty " << std::endl;
            set_flag(ExecutionFlags::SKIP);
        }

        // subgraph_input_changed can be available only shape_of is dynamic.
        // shape_of_subgraph for static shape_of could be run every inference if constant propagation does not work.
        if (_node->is_in_shape_of_subgraph() && dependant_shape_of_insts.front()->is_dynamic()) {
            bool subgraph_input_changed = false;
            for (size_t i = 0; i < dependant_shape_of_insts.size(); i++) {
                if (dependant_shape_of_insts[i]->get_flag(ExecutionFlags::SHAPE_CHANGED)) {
                    subgraph_input_changed = true;
                    break;
                }
            }
            if (!subgraph_input_changed) {
                GPU_DEBUG_TRACE_DETAIL << id() << " : Skipping execution because dependent shapeof node is not changed " << std::endl;
                set_flag(ExecutionFlags::SKIP);
            }
        }

        if (get_flag(ExecutionFlags::SKIP)) {
            update_shape_done_by_other = false; // reset
            return;
        }

        // Check successor reorder if layouts are same
        // Need to set can_be_optimized for user reorder at predecessor because
        // if the user is can_be_optimized and output node then current nodes' output should be allocated to host.
        do_runtime_skip_reorder();
        do_runtime_skip_gather();
        update_paddings();
        do_runtime_in_place_kv_cache();
        do_runtime_skip_permute();
        do_runtime_skip_strided_slice();
        do_runtime_skip_broadcast();
        do_runtime_skip_scatter_update();
        do_runtime_in_place_crop();

        if (!is_valid_fusion()) {
            OV_ITT_SCOPED_TASK(ov::intel_gpu::itt::domains::intel_gpu_plugin, openvino::itt::handle("unfused_subgraph_build: " + id()));
            get_unfused_subgraph();
            return;
        }

        // Try update impl if current impl is dynamic because opt kernel may be added to impl cache through async compilation.
        // Only try update weight and realloc when impl is updated.
        const bool can_use_async_compilation = use_async_compilation();
        const bool shape_changed = get_flag(ExecutionFlags::SHAPE_CHANGED);
        if (shape_changed || !_impl || (!shape_changed && _impl->is_dynamic() && can_use_async_compilation)) {
            update_impl(can_use_async_compilation);
            if (get_flag(ExecutionFlags::IMPL_CHANGED)) {
                update_weights();
                realloc_if_needed(prev_execution_skipped);
            }
        }

        // Paged Attention may require dispatch data update and internal buffers reallocation
        // even if the input shapes haven't been changed
        if (_node->is_type<paged_attention>() && !get_flag(ExecutionFlags::IMPL_CHANGED) && _impl->requires_update(*this, *_impl_params)) {
            _impl->update(*this, *_impl_params);

            realloc_if_needed(prev_execution_skipped);
        }

        OPENVINO_ASSERT(_impl_params->get_output_layout().is_static(),
                        "[GPU] Can't execute ", primitive_id, " primitive as output layout is dynamic in runtime");
    }
    update_shape_done_by_other = false; // reset
    OPENVINO_ASSERT(_impl != nullptr, "[GPU] Implementation is nullptr for ", primitive_id,  " primitive");

    std::function<bool(const cldnn::primitive_inst*)> has_dynamic_dependencies_insts =
        [&has_dynamic_dependencies_insts](const cldnn::primitive_inst* prim_inst) {
        for (auto& dep : prim_inst->_deps) {
            const cldnn::primitive_inst* dep_inst = dep.first;
            if (dep_inst->get_flag(ExecutionFlags::MEMORY_CHANGED)) {
                return true;
            } else if (dep_inst->can_be_optimized()) {
                if (has_dynamic_dependencies_insts(dep_inst)) {
                    return true;
                }
            }
        }
        return false;
    };

    bool need_args_update = get_flag(ExecutionFlags::IMPL_CHANGED) || get_flag(ExecutionFlags::MEMORY_CHANGED);

    // Output buffer may be changed under the following conditions, so we need to set args to kernel on each iteration
    if ((is_dynamic() && need_args_update) || has_mutable_input() || is_output() || has_dynamic_dependencies_insts(this) || _use_shared_kernels) {
        // For ocl_v2 impls we call set args based in flag in the execute() impl, so need to update the flag here
        set_flag(ExecutionFlags::MEMORY_CHANGED);
        set_arguments();
    }
    on_execute();

    if (!_node->is_type<condition>() && !_node->is_type<loop>()) {
        for (size_t i = 0; i < _outputs.size(); ++i) {
            if ((!orig_outputs[i] && _outputs[i]) || (orig_outputs[i] && !_outputs[i])) {
                set_flag(ExecutionFlags::MEMORY_CHANGED);
                break;
            }
            if (!_network.get_engine().is_the_same_buffer(*orig_outputs[i], *_outputs[i])) {
                set_flag(ExecutionFlags::MEMORY_CHANGED);
                break;
            }
        }
    }
    GPU_DEBUG_TRACE << id() << ": execute " << _impl->get_kernel_name() << " (is_dynamic=" << _impl->is_dynamic()
                    << ", "
                    << "can_be_optimized=" << can_be_optimized() << ")" << std::endl;

    const bool out_of_order_queue = get_network().get_stream().get_queue_type() == QueueTypes::out_of_order;
    if (!_exec_deps.empty()) {
        // Prepare dependencies events in case of OOO queue, CPU implementation,
        // or optimized_out impl which has CPU users (needs_completion_event() && !is_output() condition)
        if (out_of_order_queue || (_impl->is_cpu() && !can_be_optimized()) || (can_be_optimized() && needs_completion_event() && !is_output())) {
            for (auto& input : _exec_deps) {
                add_dep_event(input->get_impl_params()->out_event);
            }
        }
    }

    // Replace multiple events with single grouped event in case of barriers synchronization to prevent `_last_barrier_ev` usage as a dependency
    // event of optimized_out instance's users, which may lead to unwanted extra synchronization of CPU impls with GPU kernels
    if (_node->is_in_shape_of_subgraph() && can_be_optimized() && _impl_params->dep_events.size() > 1 && out_of_order_queue) {
        auto grouped_ev = get_network().get_stream().group_events(_impl_params->dep_events);
        _impl_params->dep_events = {grouped_ev};
    }
}

void primitive_inst::execute() {
    GPU_DEBUG_PROFILED_STAGE(instrumentation::pipeline_stage::inference);
    if (get_flag(ExecutionFlags::SKIP)) {
        set_out_event(get_network().get_stream().aggregate_events(_impl_params->dep_events));
        return;
    }

    if (_unfused_subgraph != nullptr) {
        for (auto& d : _deps) {
            if (!d.first->get_node().is_type<data>()) {
                auto allocated_mem = d.first->output_memory_ptr();
                auto actual_input_layout = d.first->get_output_layout();
                auto& engine = _network.get_engine();
                cldnn::memory_ptr actual_mem = nullptr;
                // Need to use actual layout, not the fake aligned memory layout
                if (actual_input_layout.count() != 0) {
                    actual_mem = engine.reinterpret_buffer(*allocated_mem, actual_input_layout);
                } else {
                    actual_mem = engine.allocate_memory(actual_input_layout);
                }
                _unfused_subgraph->set_input_data(d.first->id(), std::move(actual_mem));
            }
        }
        GPU_DEBUG_TRACE_DETAIL << "[Start] Executing unfused subgraph of " << id() << std::endl;
        auto outputs = _unfused_subgraph->execute(_impl_params->dep_events);
        GPU_DEBUG_TRACE_DETAIL << "[End] Finished executing unfused subgraph of " << id() << std::endl;

        auto last_fd = _impl_params->fused_desc.back();
        auto last_prim_id = last_fd.desc->id;

        OPENVINO_ASSERT(outputs.find(last_prim_id) != outputs.end(), "[GPU] Can't find output primitive ", last_prim_id, " for unfused subgraph");

        _outputs[0] = outputs.at(last_prim_id).get_memory();

        _impl_params->output_layouts[0] = _unfused_subgraph->get_output_layout(last_prim_id);
        set_out_event(outputs.at(last_prim_id).get_event());
        return;
    }

    set_out_event(_impl->execute(_impl_params->dep_events, *this));

    GPU_DEBUG_IF(!get_config().get_dump_profiling_data_path().empty()) {
        auto ev = _impl_params->out_event;
        get_network().get_stream().wait_for_events({ev});

        if (ev != nullptr) {
            auto profiling_info = ev->get_profiling_info();
            for (const auto &interval : profiling_info) {
                if (interval.stage == cldnn::instrumentation::profiling_stage::executing) {
                    GPU_DEBUG_CODE(stage_prof.set_custom_stage_duration(interval.value->value()));
                }
            }
        }
    }
}

void primitive_inst::set_arguments() {
    OV_ITT_SCOPED_TASK(ov::intel_gpu::itt::domains::intel_gpu_plugin, openvino::itt::handle("set_arguments: " + id()));
    GPU_DEBUG_PROFILED_STAGE(instrumentation::pipeline_stage::set_arguments);
    OPENVINO_ASSERT(_has_valid_input, id(), " has invalid/unset input");
    _impl->set_arguments(*this);
}

void primitive_inst::build_deps() {
    if (!_deps.empty())
        return;

    OPENVINO_ASSERT(_node != nullptr, "_node should not be nullptr for build_deps.");

    if (_deps.empty() && !_node->get_dependencies().empty()) {
        _deps = _network.get_primitives(_node->get_dependencies());
        _exec_deps = build_exec_deps(_deps);
    }
}

void primitive_inst::configure_shape_of_dependencies() {
    if (dependant_shape_of_insts.empty() && _node != nullptr) {
        for (auto shape_of : _node->get_dependant_shape_of_nodes()) {
            dependant_shape_of_insts.push_back(_network.get_primitive(shape_of->id()).get());
        }
    }
}

primitive_inst::primitive_inst(network& network)
    : _network(network)
    , _node(nullptr)
    , _impl_params(std::make_unique<kernel_impl_params>())
    , _impl(nullptr)
    , _outputs({})
    , _reordered_weights_cache(network.get_weights_cache_capacity())
    , _mem_allocated(false)
    , _type(nullptr) {}

primitive_inst::primitive_inst(network & network, program_node const& node, bool allocate_memory)
    : _network(network)
    , _node(&node)
    , _node_output_layout(node.get_output_layout())
    , _use_shared_kernels(node.get_program().get_config().get_enable_kernels_reuse())
    , _impl_params(node.get_kernel_impl_params())
    , _impl(node.get_selected_impl() ? node.get_selected_impl()->clone() : nullptr)
    , _runtime_memory_dependencies(node.get_memory_dependencies())
    , _outputs({})
    , _reordered_weights_cache(network.get_weights_cache_capacity())
    , _is_dynamic(node.is_dynamic())
    , _type(node.type())
    , _id(node.id())
    , _org_id(node.get_org_primitive_id())
    , _is_input(node.is_input())
    , _is_output(node.is_output())
    , _inputs_memory_count(node.get_inputs_count())
    , _outputs_memory_count(node.get_outputs_count())
    , _fused_mem_count(node.get_fused_inputs_count())
    , _fused_mem_offset((_fused_mem_count > 0 && node.get_first_fused_dep_idx() > 0) ? static_cast<uint64_t>(node.get_first_fused_dep_idx()) : 0)
    , _can_be_optimized(node.can_be_optimized())
    , _can_share_buffer(node.can_share_buffer())
    , _is_constant(node.is_constant())
    , _needs_completion_event(is_any_user_cpu(node.get_users()) || node.is_output()) {
    // When dynamic shape node has huge upper boundary which causes bigger mem size than system max allocable mem size, do not allocate in build time.
    auto output_layout = node.get_output_layout();
    auto& engine = network.get_engine();
    if (allocate_memory && node.is_dynamic() && (!engine.check_allocatable(output_layout, engine.get_lockable_preferred_memory_allocation_type(false)))) {
        allocate_memory = false;
    }
    _mem_allocated = allocate_memory;
    if (!_mem_allocated && (node.is_dynamic() && _outputs_memory_count > 1)) {
        auto available_allocate_memory = [&](std::vector<cldnn::layout>& layouts) -> bool {
            for (auto& l : layouts) {
                if (l.is_static())
                    return true;
            }
            return false;
        };
        allocate_memory = _mem_allocated = available_allocate_memory(_impl_params->output_layouts);
    }

    if (allocate_memory) {
        // In case when output is mutable_data primitive, and other users dependencies are only used for
        // synchronization, The output memory of such primitive will be fused with mutable_data
        auto users = node.get_users();
        auto user_count = users.size();
        uint32_t mutable_data_count = 0;
        for (auto& user : users) {
            // Get mutable_data nodes count from nodes users
            if (user->is_type<mutable_data>()) {
                mutable_data_count++;
            }
        }

        // TODO: Remove WA for arg_max_min node.
        // For now it's required to handle the case when only second output of TopK primitive is used in plugin,
        // but kernels always write both outputs to the same memory object which leads to wrong result.
        if (user_count == 1 && mutable_data_count == 1 && !node.is_type<arg_max_min>()
                                                       && !node.is_type<experimental_detectron_roi_feature_extractor>()) {
            for (auto& user : node.get_users())
                if (user->is_type<mutable_data>())
                    _outputs[0] = user->as<mutable_data>().get_attached_memory_ptr();
        } else {
            _outputs = allocate_outputs();
        }
    }
    if (_node) {
        GPU_DEBUG_TRACE_DETAIL << _node->type()->to_string(*_node) << "\n";
    }
    _impls_factory = std::make_shared<ImplementationsFactory>(_node);
    _impl_params->strm = _network.get_stream_ptr();
    for (size_t i = 0; i < get_node().get_output_layouts().size(); ++i) {
        if (_outputs.size() > i) {
            _max_output_layout_count.push_back(_outputs[i] ? _outputs[i]->get_layout().get_linear_size() : 0);
        } else {
            _outputs.push_back(nullptr);
            _max_output_layout_count.push_back(0);
        }
    }
    OPENVINO_ASSERT(_outputs.size() == get_node().get_output_layouts().size());
    OPENVINO_ASSERT(_max_output_layout_count.size() == get_node().get_output_layouts().size());
}

memory::ptr primitive_inst::allocate_internal_buffer(const layout& layout, size_t idx, bool reset, bool lockable) {
    if (_impl == nullptr || _outputs.empty() || _outputs[0] == nullptr)
        return nullptr;

    auto device_mem_acc = [&](size_t a, std::pair<primitive_inst*, int32_t> b) {
        if (!b.first->mem_allocated()) return a;
        auto res = a;
        for (size_t i = 0; i < b.first->outputs_memory_count(); ++i) {
            if (b.first->output_memory(i).get_allocation_type() == allocation_type::usm_device ||
                b.first->output_memory(i).get_allocation_type() == allocation_type::cl_mem)
                return a + b.first->output_memory().size();
        }
        return res;
    };

    auto& engine = get_network().get_engine();
    bool input_device_mem = false;

    // NOTE: Currently the ocl driver aborts at runtime when there are layers using device memory close to max size within multiple streams.
    // Decided the limitation as 85 % empirically, but still it needs further investigation.
    const auto& inst_deps = _network.get_primitives(_node->get_dependencies());

    auto total_device_mem_size = std::accumulate(inst_deps.begin(), inst_deps.end(), size_t(0), device_mem_acc);
    for (const auto& output : _outputs) {
        if (output->get_allocation_type() == allocation_type::usm_device)
            total_device_mem_size += output->size();
    }

    int64_t available_device_mem_size = engine.get_device_info().max_global_mem_size - total_device_mem_size;
    // TODO: check if this logic is needed
    // check if there is any device mem input
    if (engine.supports_allocation(allocation_type::usm_device)) {
        for (const auto& dep : inst_deps) {
            if (dep.first->output_memory_ptr() &&
                dep.first->output_memory_ptr()->get_allocation_type() == allocation_type::usm_device) {
                input_device_mem = true;
                break;
            }
        }
    }
    // allocate intermediate memory for the updated layout of buffer
    auto alloc_type = allocation_type::unknown;
    GPU_DEBUG_LOG << "[" << _node->id() << ": internal buf " << idx << "] "
                  << layout.to_short_string() << " lockable=" << lockable << std::endl;
    if ((int64_t)available_device_mem_size - (int64_t)layout.bytes_count() >= 0 &&
        (input_device_mem || _node->get_preferred_impl_type() == impl_types::onednn) && !lockable) {
        // scratchpad memory type enforces to device mem.
        GPU_DEBUG_LOG << " input is device mem and available device mem size (" << available_device_mem_size
                      << ") > requested memory (" << layout.bytes_count() << " )" << std::endl;
        alloc_type = engine.get_preferred_memory_allocation_type();
    } else {
        GPU_DEBUG_LOG << " input is not device mem or available device mem size ("
                      << available_device_mem_size << ") <= requested memory (" << layout.bytes_count() << " )" << std::endl;
        alloc_type = engine.get_lockable_preferred_memory_allocation_type();
    }
    GPU_DEBUG_LOG << "=> allocate to " << alloc_type << std::endl;

    // Reuse intermediate buffer like output buffer.
    bool reuse_internal_buf = true;
    auto ret_mem =
        get_memory_from_pool(_network.get_engine(),
                             _network.get_id(),
                             _network.get_memory_pool(),
                             *_node,
                             layout,
                             alloc_type,
                             reuse_internal_buf,
                             _runtime_memory_dependencies,
                             reset,
                             _intermediates_memory.size() > idx ? _intermediates_memory[idx].get() : nullptr);
    GPU_DEBUG_LOG << " [" << _network.get_id() << ":" << _node->id() << ": internal buf " << idx << "] " << alloc_type
                  << " " << ret_mem->buffer_ptr() << std::endl;
    return ret_mem;
}

void primitive_inst::allocate_internal_buffers(bool reset) {
    if (_impl == nullptr || _outputs.empty() || _outputs[0] == nullptr)
        return;
    const auto& buffer_descs = _impl->get_internal_buffer_descs(*_impl_params);
    if (buffer_descs.empty())
        return;

    // allocate intermediate memory for the updated layout of buffer
    std::vector<memory::ptr> intermediates_memory;
    for (size_t i = 0; i < buffer_descs.size(); ++i) {
        if (buffer_descs[i].m_layout.get_linear_size() == 0)
            continue;
        intermediates_memory.push_back(allocate_internal_buffer(buffer_descs[i].m_layout, i, reset));
        max_intermediates_memory_sizes.push_back(intermediates_memory[i]->size());
    }
    _intermediates_memory = intermediates_memory;
}

void primitive_inst::update_weights() {
    OV_ITT_SCOPED_TASK(ov::intel_gpu::itt::domains::intel_gpu_plugin, openvino::itt::handle("update_weights: " + id()));
    GPU_DEBUG_PROFILED_STAGE(instrumentation::pipeline_stage::update_weights);
    if (!_impl)
        return;

    bool weightable_node = _node->is_type<fully_connected>() || _node->is_type<convolution>() || _node->is_type<deconvolution>();
    if (!weightable_node)
        return;

    auto& engine = _network.get_engine();
    auto reorder_kernel_params = _impl->get_weights_reorder_kernel_params();

    if (reorder_kernel_params)
        reorder_kernel_params->prog = get_network().get_program().get();

    auto weights_idx = _node->get_primitive()->input.size();
    auto original_weights_memory = dep_memory_ptr(weights_idx);
    const auto& original_layout = original_weights_memory->get_layout();

    if (!reorder_kernel_params) {
        // If kernel doesn't says that it doesn't require weights reorder, but weights were reordered previously, then
        // incorrect memory buffer may be assigned, so reset cached weights for such case
        _reordered_weights_cache.add(original_layout, original_weights_memory);
        _impl_params->weights_layout = optional_layout(original_layout);
        GPU_DEBUG_TRACE_DETAIL << id() << ": add original weights memory " << original_layout.to_short_string() << " to weights cache; "
                                       << "cache_size=" << _reordered_weights_cache.size() << "/" << _reordered_weights_cache.capacity() << std::endl;
    } else {
        // Set original partial shape, because it may be lost during kernel_selector::weights_tensor -> layout conversion
        auto expected_layout =
            reorder_kernel_params->get_output_layout().clone_with_other_shape(original_layout.get_partial_shape());
        _impl_params->weights_layout = optional_layout(expected_layout);

        if (_reordered_weights_cache.has(expected_layout)) {
            GPU_DEBUG_PROFILED_STAGE_CACHE_HIT(true);
            GPU_DEBUG_TRACE_DETAIL << id() << ": reuse weights for " << expected_layout.to_short_string() << std::endl;
            return;
        } else if (original_layout.compatible(expected_layout)) {
            GPU_DEBUG_PROFILED_STAGE_CACHE_HIT(true);
            GPU_DEBUG_TRACE_DETAIL << id() << ": reinterpret original weights memory from " << original_layout.to_short_string()
                                           << " to " << expected_layout.to_short_string() << std::endl;
            _reordered_weights_cache.add(expected_layout, engine.reinterpret_buffer(*original_weights_memory, expected_layout));
            return;
        } else {
            GPU_DEBUG_PROFILED_STAGE_CACHE_HIT(false);
            auto& cache = get_network().get_program()->get_implementations_cache();
            auto reorder_inst = std::make_shared<cldnn::reorder_inst>(get_network());

            if (auto cached_impl = cache.get(*reorder_kernel_params)) {
                GPU_DEBUG_TRACE_DETAIL << id() << ": reorder weights (cached) from " << original_layout.to_short_string()
                                       << " to " << expected_layout.to_short_string() << std::endl;
                reorder_inst->set_impl(cached_impl->clone());
            } else {
                GPU_DEBUG_TRACE_DETAIL << id() << ": reorder weights from " << original_layout.to_short_string()
                                       << " to " << expected_layout.to_short_string() << std::endl;

                auto impl_type = (reorder_kernel_params->get_output_layout(0).format == format::custom) ? impl_types::onednn : impl_types::ocl;
                auto factory = reorder::type_id()->get_best_impl(impl_type, shape_types::static_shape);
                auto reorder_impl = factory->create(*reorder_kernel_params);
                if (impl_type == impl_types::ocl) {
                    auto& kernels_cache = get_network().get_program()->get_kernels_cache();
                    auto kernels = kernels_cache.compile(*reorder_kernel_params, reorder_impl->get_kernels_source());
                    OPENVINO_ASSERT(kernels.size() == 1, "[GPU] Expected number of compiled kernels is 1, but got ", kernels.size());
                    reorder_impl->set_kernels(kernels);
                    reorder_impl->can_share_kernels = _use_shared_kernels;
                }

                reorder_inst->set_impl(reorder_impl->clone());

                cache.add(*reorder_kernel_params, reorder_impl->clone());
            }

            auto& stream = get_network().get_stream();

            bool can_reuse = false;
            memory::ptr weights_memory = nullptr;
            if (_reordered_weights_cache.is_full()) {
                weights_memory = _reordered_weights_cache.get_lru_element().second;
                can_reuse = weights_memory->size() <= expected_layout.bytes_count() && (weights_memory->buffer_ptr() != original_weights_memory->buffer_ptr());
            }

            if (can_reuse) {
                GPU_DEBUG_TRACE_DETAIL << id() << ": reuse weights memory for new layout " << expected_layout.to_short_string() << std::endl;
                weights_memory = engine.reinterpret_buffer(*weights_memory, expected_layout);
            } else {
                GPU_DEBUG_TRACE_DETAIL << id() << ": allocate weights memory" << std::endl;
                auto alloc_type = engine.get_preferred_memory_allocation_type();
                weights_memory = engine.allocate_memory(expected_layout, alloc_type);
            }

            _reordered_weights_cache.add(expected_layout, weights_memory);
            GPU_DEBUG_TRACE_DETAIL << id() << ": update weights cache: " << expected_layout.to_short_string() << " cache_size="
                                   << _reordered_weights_cache.size() << "/" << _reordered_weights_cache.capacity() << std::endl;

            kernel_arguments_data args;
            args.inputs.push_back(original_weights_memory);
            args.outputs.push_back(weights_memory);

            auto reorder_impl = reorder_inst->get_impl();
            reorder_impl->set_arguments(*reorder_inst, args);
            add_dep_event(reorder_impl->execute({}, *reorder_inst));

            GPU_DEBUG_IF(!get_config().get_dump_profiling_data_path().empty()) {
                stream.wait_for_events(_impl_params->dep_events);
            }

            return;
        }
    }

    GPU_DEBUG_PROFILED_STAGE_CACHE_HIT(true);

    return;
}

static bool user_requesting_mem_reuse_false(const program_node& node) {
    for (auto& user : node.get_users()) {
        if ((user->get_selected_impl() != nullptr) && (user->get_selected_impl()->can_reuse_memory == false)) {
            return true;
        } else if (user->get_selected_impl() == nullptr) {
            if (user->is_dynamic()) {
                return true;
            }
            if (user_requesting_mem_reuse_false(*user)) {
                return true;
            }
        }
    }
    return false;
}

memory::ptr primitive_inst::allocate_output(engine& _engine,
                                            memory_pool& pool,
                                            const program_node& _node,
                                            const kernel_impl_params& impl_params,
                                            const std::unordered_set<size_t>& memory_dependencies,
                                            uint32_t net_id,
                                            bool is_internal,
                                            size_t idx,
                                            bool reset,
                                            bool is_output_buffer,
                                            memory* curr_memory,
                                            bool runtime_alloc) {
    const auto& out_layout = impl_params.get_output_layout(idx);
    OPENVINO_ASSERT(out_layout.is_static() || out_layout.has_upper_bound(), "[GPU] Can't allocate output for dynamic layout");
    auto device_mem_acc = [&](size_t a, const cldnn::layout& l) {
        // Input shape may be dynamic is some cases (shape_of). It means that output shape of node doesn't depend on input shape
        // and out memory can be allocated on program build stage.
        if (l.is_static())
            return a + l.bytes_count();

        return a;
    };

    const auto& device_info = _engine.get_device_info();
    auto layout = out_layout.clone_with_other_shape(out_layout.get_partial_shape().get_max_shape());
    bool usm_device_allocatable = true;
    const auto& total_device_input_mem_size = std::accumulate(impl_params.input_layouts.begin(), impl_params.input_layouts.end(), (uint64_t)0, device_mem_acc);
    if (total_device_input_mem_size > device_info.max_global_mem_size)
        usm_device_allocatable = false;

    bool reusable_across_network = (runtime_alloc && _node.is_dynamic_output_layout())
                                    || !user_requesting_mem_reuse_false(_node);

    // Do not use memory pool for nodes from shape_of subgraphs, because such nodes mostly use CPU impls and may be executed in parallel with predecessors
    // GPU kernels and cause accuracy problems. This significantly improves performance (because provides an ability not to synchronize shape_of subgraphs
    // execution with other nodes) at the cost of tiny increase in memory consumption.
    if (_node.is_in_shape_of_subgraph())
        reusable_across_network = false;

    if (reusable_across_network && _node.get_program().is_body_program() && is_output_buffer && runtime_alloc)
        reusable_across_network = false;

    // For outputs, cpu prim we want to have lockable alloc type
    // Also if the successor of a node is an cpu, then memory needs to be lockable.
    bool is_cpu = _node.get_selected_impl() ? _node.get_selected_impl()->is_cpu() :
                                              _node.get_preferred_impl_type() == impl_types::cpu;

    auto total_output_bytes = layout.bytes_count();
    auto use_lockable_memory =
        (is_output_buffer && ov::intel_gpu::can_use_usm_host(_engine, total_output_bytes)) ||
        is_cpu || has_any_cpu_user_not_shape_of(_node.get_users()) ||
        !_engine.supports_allocation(allocation_type::usm_device) ||
        (_node.is_shape_infer_dep() && device_info.dev_type == device_type::integrated_gpu);
    const auto& lockable_mem_type = _engine.get_lockable_preferred_memory_allocation_type(layout.format.is_image_2d());

    auto alloc_type = use_lockable_memory ? lockable_mem_type
                    : !usm_device_allocatable ? lockable_mem_type : allocation_type::usm_device;

    if (is_internal) {
        bool is_reorder_weights = _node.is_type<reorder>() && _node.as<reorder>().get_primitive()->weights_reorder_params;
        if (_node.can_be_optimized() || is_reorder_weights) {
            GPU_DEBUG_LOG << "[" << _node.id() << ": output]" << std::endl;
            // Use usm_device memory for weights reordering
            if (is_internal && is_reorder_weights &&
                _engine.supports_allocation(allocation_type::usm_device))
                alloc_type = allocation_type::usm_device;
            return get_memory_from_pool(_engine,
                                        net_id,
                                        pool,
                                        _node,
                                        layout,
                                        alloc_type,
                                        false,
                                        memory_dependencies,
                                        reset,
                                        curr_memory);
        } else {
            if ((_node.is_output() && is_reorder_weights) || (!_node.is_output() && _node.is_type<input_layout>()))
                reset = false;
            GPU_DEBUG_LOG << "[" << _node.id() << ": constant]" << std::endl;
            return _engine.allocate_memory(layout, alloc_type, reset);
        }
    } else if (!_node.can_share_buffer() || impl_params.can_be_optimized() || _node.is_output()) {
        GPU_DEBUG_LOG << "[" << _node.id() << ": output]" << std::endl;
        return _engine.allocate_memory(layout, alloc_type, reset);
    } else {
        return get_memory_from_pool(_engine,
                                    net_id,
                                    pool,
                                    _node,
                                    layout,
                                    alloc_type,
                                    reusable_across_network,
                                    memory_dependencies,
                                    reset,
                                    curr_memory);
    }
}

std::vector<memory::ptr> primitive_inst::allocate_outputs(kernel_impl_params* updated_params, bool reset_mem, bool runtime_alloc) {
    std::vector<memory::ptr> outputs;
    outputs.reserve(get_node().get_outputs_count());
    const auto& impl_params = updated_params != nullptr ? *updated_params : *_impl_params;
    const auto& out_layouts = impl_params.output_layouts;
    set_flag(ExecutionFlags::MEMORY_CHANGED);
    for (size_t i = 0; i < get_node().get_outputs_count(); ++i) {
        if (out_layouts[i].is_dynamic() && !out_layouts[i].has_upper_bound()) {
            outputs.push_back(memory::ptr());
        } else {
            auto current_memory_ptr = _outputs.size() > i ? output_memory_ptr(i).get() : nullptr;
            auto is_output = is_output_buffer(this, runtime_alloc);

            outputs.push_back(allocate_output(_network.get_engine(),
                                            _network.get_memory_pool(),
                                            *_node,
                                            impl_params,
                                            _runtime_memory_dependencies,
                                            get_network_id(),
                                            _network.is_internal(),
                                            i,
                                            reset_mem,
                                            is_output,
                                            current_memory_ptr,
                                            runtime_alloc));
        }
    }
    return outputs;
}

std::vector<primitive_inst*> primitive_inst::build_exec_deps(std::vector<std::pair<primitive_inst*, int32_t>> const& deps) {
    std::vector<primitive_inst*> exec_deps;
    exec_deps.reserve(deps.size());
    for (auto& dep : deps)
        if (dep.first->get_impl() != nullptr || dep.first->is_dynamic())
            exec_deps.push_back(dep.first);

    return exec_deps;
}

std::string primitive_inst::generic_to_string(program_node const& node, const char* type_name) {
    auto node_info = node.desc_to_json();

    std::stringstream primitive_description;
    std::stringstream ss_inputs;

    for (size_t i = 0; i < node.get_dependencies().size(); ++i) {
        auto& in = node.get_dependency(i);
        ss_inputs << in.id();
        ss_inputs << ", count: " << in.get_output_layout().count();
        i != (node.get_dependencies().size() - 1) ? ss_inputs << ", " : ss_inputs << "";
    }

    json_composite generic_info;
    generic_info.add("type_name", type_name);
    generic_info.add("deps count", node.get_dependencies().size());
    generic_info.add("deps", ss_inputs.str());

    node_info->add("generic info", generic_info);
    node_info->dump(primitive_description);

    return primitive_description.str();
}

cldnn::network::ptr primitive_inst::get_unfused_subgraph() {
    GPU_DEBUG_TRACE_DETAIL << id() << ": Use unfused subgraph due to unexpected fusions\n";
    if (!_unfused_subgraph) {
        topology t;

        std::vector<primitive_id> outer_dep_ids;
        // Add input primitives: constants are moved as is
        // Any other primitive types are replaced with input_layout
        auto prim_of_fused_node = std::const_pointer_cast<primitive>(_impl_params->desc);
        size_t dep_idx = 0;
        for (auto& dep : _node->get_dependencies()) {
            cldnn::primitive_id dep_id = dep.first->id();

            if (dep.first->is_type<data>()) {
                auto& data_node = dep.first->as<data>();
                // need to rename primitive ids of dependent data of the current fused nodes with those in the original primitive
                if (dep_idx >= prim_of_fused_node->input_size() && dep_idx < prim_of_fused_node->dependencies().size())
                    dep_id = prim_of_fused_node->dependencies()[dep_idx].pid;
                // mem field of original primitive can be nullified during transfer_memory_to_device pass, thus use mem from program_node
                data data_prim(dep_id, data_node.get_attached_memory_ptr());
                t.add(data_prim);
            } else {
                input_layout in_prim(dep_id, dep.first->get_output_layout());
                t.add(in_prim);
            }
            outer_dep_ids.push_back(dep_id);
            dep_idx += 1;
        }

        // Create the primitive itself
        t.add_primitive(std::const_pointer_cast<primitive>(_node->get_primitive()));
        outer_dep_ids.push_back(_node->id());

        // Add primitives for fused-ops
        for (auto& fd : _impl_params->fused_desc) {
            auto prim = std::const_pointer_cast<primitive>(fd.desc);
            for (size_t i = 0; i < prim->input.size(); i++) {
                auto& in = prim->input[i];
                // If dependency name is not found in current topology, we need to remap it
                // It may happen if dependency primitive has been fused into some previous primitive, e.g:
                // prim1 -> eltwise1 -> eltwise2
                //          prim2 -------/
                //  fused_prim1=prim1 + eltwise1
                //  fused_prim2=prim2 + eltwise2
                // from the names perspective fused graph will looka as follows:
                // prim1 -> prim2
                // And when we construct unfused subgraph for prim2, we take original eltwise2 primitive which expects eltwise1 primitive as input
                // which doesn't exist anymore in the graph
                // Thus we update dependency name used dependencies idx stored in fused descriptor.
                if (std::find_if(outer_dep_ids.begin(), outer_dep_ids.end(), [&](const primitive_id& pid) {
                        return pid == in.pid;
                    }) == outer_dep_ids.end()) {
                    if (fd.has_outer_dep()) {
                        size_t dep_id = fd.outer_dep_start_idx;
                        auto outer_dep_id = _node->get_dependency(dep_id).id();

                        if (std::find_if(fd.deps.begin(), fd.deps.end(), [&](const std::pair<cldnn::primitive_id, size_t>& dep_info) {
                                return (dep_info.first == outer_dep_id && dep_info.second == i);
                            }) == fd.deps.end()) {
                            in = _node->id();
                        } else {
                            in = outer_dep_id;
                        }
                    } else {
                        in = _node->id();
                    }
                }
            }
            t.add_primitive(prim);
            outer_dep_ids.push_back(prim->id);
        }
        // Samely, need to update dependency of the current fused nodes' input primitive ids with those in the current program
        for (size_t i = 0; i < prim_of_fused_node->input.size(); ++i) {
            auto& in = prim_of_fused_node->input[i];
            if (std::find_if(outer_dep_ids.begin(), outer_dep_ids.end(),
                             [&](const primitive_id& pid) {
                                 return pid == in.pid;
                             }) == outer_dep_ids.end()) {
                in = _node->get_dependency(i).id();
            }
        }
        ExecutionConfig subgraph_config{
            ov::intel_gpu::allow_static_input_reorder(true),
            ov::intel_gpu::allow_new_shape_infer(true),
            ov::enable_profiling(get_network().get_config().get_enable_profiling()),
            ov::intel_gpu::use_onednn(get_network().get_config().get_use_onednn())
        };
        auto prog = program::build_program(get_network().get_engine(),
                                           t,
                                           subgraph_config,
                                           get_network().get_program()->get_task_executor(),
                                           get_network().get_program()->get_compilation_context_ptr(),
                                           true,
                                           false);

        _unfused_subgraph = network::allocate_network(get_network().get_stream_ptr(), prog, true, get_network().is_primary_stream());
    }
    return _unfused_subgraph;
}

bool primitive_inst::is_valid_fusion() const {
    if (!is_dynamic())
        return true;

    const auto& fuse_descriptors = _impl_params->fused_desc;
    if (fuse_descriptors.empty())
        return true;
    std::vector<fused_primitive_desc> fused_eltwise_prims;
    for (auto& fd : fuse_descriptors) {
        if (fd.is_type<eltwise>() || fd.is_type<activation>()) {
            fused_eltwise_prims.push_back(fd);
        } else {
            if (fd.is_type<reorder>() || fd.is_type<quantize>())
                continue;
            if (fd.is_type<swiglu>()) {
                OPENVINO_ASSERT(_node->is_type<fully_connected>() && _node->get_preferred_impl_type() == impl_types::ocl);
                if (!_node->get_selected_impl())
                    return false;
                // TODO : support ref kernel too
                if (_node->get_selected_impl()->get_kernel_name().find("fully_connected_gpu_bf_tiled") != std::string::npos)
                    return true;
                else
                    return false;
            }

            OPENVINO_THROW("[GPU] Unsupported fused operation in dynamic shape: type=", fd.desc->type_string(), ", id=", fd.desc->id);
        }
    }

    if (fused_eltwise_prims.empty())
        return true;

    if (_node->is_type<fully_connected>() || _node->is_type<gemm>() || _node->is_type<convolution>()) {
        if (_impl_params->input_layouts[0].count() == 0 ||
            _impl_params->input_layouts[1].count() == 0) {
            return false;
        }
    }

    if (_node->is_type<fully_connected>() && _node->get_preferred_impl_type() == impl_types::ocl) {
        // TODO: Only fc_bf_tiled_kernel & ref kernel are verified for fused eltwise. To support more fc kernels for eltwise fusion
        if (!_node->get_selected_impl())
            return false;
        if (!data_type_traits::is_i8_u8(_node->get_input_layout(0).data_type) &&
            (_node->get_selected_impl()->get_kernel_name().find("fully_connected_gpu_bf_tiled") == std::string::npos) &&
            (_node->get_selected_impl()->get_kernel_name().find("fully_connected_gpu_bfyx_ref") == std::string::npos)) {
            return false;
        }
    }

    const auto& out_pshape = _impl_params->get_output_layout().get_partial_shape();
    for (auto& fd : fused_eltwise_prims) {
        auto outer_dep_idx = fd.outer_dep_start_idx;
        if (outer_dep_idx < 0) // no outer dep
            continue;
        OPENVINO_ASSERT(fd.total_num_deps == 2, "[GPU] Unexpected count of dependencies in dynamic fusion for eltwise or activation");
        OPENVINO_ASSERT(outer_dep_idx < 0 || static_cast<int32_t>(_deps.size()) > outer_dep_idx, "[GPU] Invalid fused dependency idx");
        const auto& outer_dep = _deps[outer_dep_idx];

        const auto& outer_dep_pshape = outer_dep.first->_impl_params->get_output_layout().get_partial_shape();
        auto merged_shape = out_pshape;
        bool can_broadcast = true;
        if (fd.is_type<eltwise>())
            can_broadcast = ov::PartialShape::broadcast_merge_into(merged_shape, outer_dep_pshape, fd.typed_desc<eltwise>()->broadcast_spec);

#ifdef ENABLE_ONEDNN_FOR_GPU
        // WA for OneDNN binary add fusions: we need to broadcast batch dimension to avoid situation with
        // batch dimension mismatch in OneDNN tensor descriptors as follow:
        // * Gemm output shape: (b,f,y,x) -> OneDNN shape: (b*f,y,x)
        // * Gemm fused op shape: (1,f,y,x) -> OneDNN shape: (1*f,y,x)
        // If batch dimension of gemm output is not equal to 1, then OneDNN will not be able to broadcast fused op data
        // correctly and we need to do it manually
        if (_node->is_type<gemm>() && _node->get_preferred_impl_type() == impl_types::onednn) {
            const auto& gemm_layout = _impl_params->get_output_layout();
            const auto& data_layout = outer_dep.first->_impl_params->get_output_layout();
            auto gemm_dims = onednn::convert_gemm_tensor(gemm_layout.get_tensor(),
                                                         cldnn::format::dimension(gemm_layout.format),
                                                         false);
            auto data_dims = onednn::convert_gemm_tensor(data_layout.get_tensor(),
                                                         cldnn::format::dimension(data_layout.format),
                                                         false);

            if (gemm_dims[0] != data_dims[0] && gemm_dims[1] != 1)
                return false;
        } else if (_node->is_type<fully_connected>() && _node->get_preferred_impl_type() == impl_types::onednn) {
            const auto& fc_layout = _impl_params->get_output_layout();
            const auto& data_layout = outer_dep.first->_impl_params->get_output_layout();

            const auto fc_dims = fc_layout.get_dims();
            const auto data_dims = data_layout.get_dims();

            auto same_spatial = [](layout a, layout b) {
                if (a.get_spatial_rank() != b.get_spatial_rank())
                    return false;
                for (size_t i = 0; i < a.get_spatial_rank(); i++) {
                    if (a.spatial(i) != b.spatial(i))
                        return false;
                }
                return true;
            };

            if (!(fc_dims[0] == 1 || fc_dims[1] == 1) &&
                !(data_dims[0] == 1 && data_dims[1] == 1) &&
                !((data_dims[0] == 1 || data_dims[1] == 1) && same_spatial(fc_layout, data_layout)) &&
                !(fc_layout.count() == data_layout.count())) {
                return false;
            }
        }
#endif

        // We check that broadcasting of extra input is possible and it doesn't change output shape. If it output shape is changed, then
        // some dimension of dep_pshape is greater than out_pshape
        if (!can_broadcast || merged_shape != out_pshape)
            return false;
    }

    return true;
}

void primitive_inst::add_profiling_data(instrumentation::pipeline_stage stage, bool cache_hit, std::string memalloc_info, int64_t time, bool per_iter_mode) {
    instrumentation::perf_counter_key key {
            _network.get_input_layouts(),
            _impl_params->input_layouts,
            _impl_params->output_layouts,
            get_implementation_name(),
            stage,
#ifdef GPU_DEBUG_CONFIG
            per_iter_mode ? get_network().get_current_iteration_num() : 0,
#else
            0,
#endif
            cache_hit,
            memalloc_info
    };

    auto hash = instrumentation::perf_counter_hash()(key);
    auto& d = _profiling_data[hash];
    if (_profiling_info.find(hash) == _profiling_info.end()) {
        _profiling_info.emplace(hash, key);
    }

    auto& total_time = std::get<0>(d);
    auto& total_iter = std::get<1>(d);
    total_time += time;
    total_iter++;
}

std::string primitive_inst::get_implementation_name() const {
    try {
        auto kernel_name = _impl ? _impl->get_kernel_name() : "";
        return !kernel_name.empty() ? kernel_name : "undef";
    } catch (...) { }

    return "undef";
}


ImplementationsFactory::ImplementationsFactory(const program_node* node)
    : m_node(node)
    , m_available_impls(node->type()->get_supported_implementations(*node))
    , m_static_impls_cache(node->get_program().get_implementations_cache())
    , m_dynamic_impls_cache() {
    if (node->get_selected_impl() && node->get_selected_impl()->is_dynamic()) {
        m_dynamic_impls_cache.emplace_back(node->get_selected_impl()->clone());
    }
}

std::shared_ptr<primitive_impl> ImplementationsFactory::get_primitive_impl_for_params(primitive_inst& inst,
                                                                                      const kernel_impl_params& params,
                                                                                      bool use_async_compilation) {
    auto find_impl = [this](const program_node* node, const kernel_impl_params& params, shape_types shape_type) -> std::unique_ptr<primitive_impl> {
        OPENVINO_ASSERT(node != nullptr);
        for (auto& impl_manager : m_available_impls) {
            if ((impl_manager->get_shape_type() & shape_type) != shape_type)
                continue;

            if (!impl_manager->support_shapes(params))
                continue;

            return impl_manager->create(*node, params);
        }

        return nullptr;
    };

    const auto node = &inst.get_node();
    auto& prog = *inst.get_network().get_program();
    auto& kernels_cache = prog.get_kernels_cache();

    // Update param if fake_alignment is available
    auto updated_params = inst.get_fake_aligned_params_if_possible(params);
    // Change weights layout of `updated_params` to original one to have valid information
    // in _impl->_weights_reorder_params about required weights format after impl selection
    if (inst.get_node().is_type<fully_connected>() || inst.get_node().is_type<convolution>() || inst.get_node().is_type<deconvolution>()) {
        const auto weights_idx = inst.get_node().get_primitive()->input.size();
        const auto original_weights_memory = inst.dep_memory_ptr(weights_idx);
        updated_params.weights_layout = optional_layout(original_weights_memory->get_layout());
    }

    for (auto& i : updated_params.input_layouts) {
        i.data_padding._dynamic_dims_mask = padding::EMPTY_MASK;
    }
    for (auto& o : updated_params.output_layouts) {
        o.data_padding._dynamic_dims_mask = padding::EMPTY_MASK;
    }

    // 1. If we have static impl in the cache - use it
    if (use_async_compilation && ((inst.get_impl() && inst.get_impl()->is_dynamic()) || inst.get_flag(ExecutionFlags::SHAPE_CHANGED))) {
        auto cached_impl = m_static_impls_cache.get(updated_params);
        if (cached_impl) {
            return cached_impl->clone();
        }

        // 1.1. Static impl not found - run async compilation
        auto& compilation_context = prog.get_compilation_context();
        compilation_context.push_task(updated_params, [&inst, &compilation_context, updated_params, find_impl]() {
            if (compilation_context.is_stopped())
                return;
            auto& _program = *inst.get_network().get_program();
            auto& cache = _program.get_implementations_cache();
            {
                // Check existense in the cache one more time as several iterations of model execution could happens and multiple compilation
                // tasks created for same shapes
                if (cache.has(updated_params))
                    return;
            }

            std::unique_ptr<primitive_impl> impl = find_impl(&inst.get_node(), updated_params, shape_types::static_shape);

            if (impl->get_kernels_source().size() > 0) {
                auto kernels = _program.get_kernels_cache().compile(updated_params, impl->get_kernels_source());
                impl->set_kernels(kernels);
            }
            cache.add(updated_params, std::move(impl));
        });
    }

    std::shared_ptr<primitive_impl> dynamic_impl = nullptr;
    // 2. Try to find existing dynamic impl which supports given shapes
    for (auto& impl : m_dynamic_impls_cache) {
        if (impl->m_manager->support_shapes(params)) {
            dynamic_impl = impl;
            break;
        }
    }

    // 3. Try to create new shape agnostic impl & cache it
    if (!dynamic_impl) {
        dynamic_impl = find_impl(node, params, shape_types::dynamic_shape);
        if (dynamic_impl && !inst.can_be_optimized()) {
            dynamic_impl->set_node_params(*node);
            auto kernels = kernels_cache.compile(params, dynamic_impl->get_kernels_source());
            dynamic_impl->set_kernels(std::move(kernels));
            m_dynamic_impls_cache.push_back(dynamic_impl);
        }
    }

    // 4. If we have any dynamic impl, do adjustment for new shape before returning in back
    if (dynamic_impl) {
        dynamic_impl->update(inst, params);
        return dynamic_impl;
    }

    // 5. Finally, if no impl found so far, we just enforce static impl compilation
    auto static_impl = find_impl(node, updated_params, shape_types::static_shape);
    assert(static_impl != nullptr);
    static_impl->set_node_params(*node);
    if (!inst.can_be_optimized()) {
        auto& kernels_cache = prog.get_kernels_cache();
        auto kernels = kernels_cache.compile(updated_params, static_impl->get_kernels_source());
        static_impl->set_kernels(std::move(kernels));
        m_static_impls_cache.add(updated_params, static_impl->clone());
    }

    return static_impl;
}

bool ImplementationsFactory::has(impl_types impl_type) const {
    return std::any_of(m_available_impls.begin(), m_available_impls.end(), [&impl_type](const std::shared_ptr<ImplementationManager>& m) {
        return m->get_impl_type() == impl_type;
    });
}

}  // namespace cldnn
