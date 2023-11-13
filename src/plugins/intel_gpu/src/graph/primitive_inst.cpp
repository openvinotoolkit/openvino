// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "program_helpers.h"
#include "primitive_inst.h"
#include "data_inst.h"
#include "mutable_data_inst.h"
#include "reorder_inst.h"
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
#include "deconvolution_inst.h"
#include "shape_of_inst.h"
#include "softmax_inst.h"
#include "gemm_inst.h"
#include "assign_inst.h"
#include "read_value_inst.h"
#include "condition_inst.h"
#include "experimental_detectron_roi_feature_extractor_inst.hpp"
#include "implementation_map.hpp"
#include "graph_optimizer/prepare_buffer_fusing.h"

#include "intel_gpu/plugin/common_utils.hpp"
#include "intel_gpu/graph/network.hpp"
#include "intel_gpu/graph/serialization/set_serializer.hpp"
#include "intel_gpu/runtime/engine.hpp"
#include "intel_gpu/runtime/memory.hpp"
#include "intel_gpu/runtime/error_handler.hpp"
#include "intel_gpu/runtime/debug_configuration.hpp"
#include "intel_gpu/runtime/compilation_context.hpp"

#include "json_object.h"
#include <string>
#include <stack>
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
            if (is_optimized_output_user<const std::shared_ptr<primitive_inst>>(user)) {
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
        return false;
    }
    if (auto impl = user->get_selected_impl())
        return impl->is_cpu();
    return false;
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
                                const std::set<std::string>& memory_dependencies,
                                bool reset = true,
                                memory* curr_memory = nullptr) {
    OPENVINO_ASSERT(!layout.is_dynamic() || layout.has_upper_bound(),
                    "[GPU] Can't allocate output for dynamic layout without upper bound");
    // Use layout with max tensor for dynamic shape with upper bound
    if (_node.get_program().get_config().get_property(ov::intel_gpu::enable_memory_pool)) {
        if (curr_memory != nullptr)
            pool.release_memory(curr_memory, _node.id(), net_id);
        return pool.get_memory(layout, _node.id(), net_id, memory_dependencies, type, reusable_across_network, reset);
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

void primitive_inst::check_memory_to_set(const memory& mem, const layout& layout) const {
    OPENVINO_ASSERT((mem.get_layout() == layout) || layout.is_dynamic(), "[GPU] Unexpected layout of input memory for ", id(), " node!\n",
                     "Node layout: ", layout.to_short_string(), "\n",
                     "Memory layout: ", mem.get_layout().to_short_string());

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
            OPENVINO_ASSERT(layout.format.is_image_2d(), "Attempt to set user-supplied input or output image instead of a buffer");
            break;
        case shared_mem_type::shared_mem_buffer:
        case shared_mem_type::shared_mem_dxbuffer:
            OPENVINO_ASSERT(!layout.format.is_image_2d(), "Attempt to set user-supplied input or output buffer instead of an image");
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
        return get_network().get_stream().create_user_event(true);
    }

    auto ol = _impl_params->get_output_layout(idx);

    if (check)
        check_memory_to_set(*mem_new, ol);

    if (is_constant()) {
        ev = mem_new->copy_from(_network.get_stream(), *_outputs[idx], false);
    } else {
        ev = get_network().get_stream().create_user_event(true);
        _outputs[idx] = mem_new;
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
            GPU_DEBUG_TRACE_DETAIL << id() << ": update shape dep: " << _deps[i].first->id()
                                   << " was: " << _impl_params->get_input_layout(i).to_short_string()
                                   << " now: " << new_shape.to_short_string() << std::endl;
            _impl_params->input_layouts[i] = new_shape;
            input_shape_changed = true;
        }
    }

    if (input_shape_changed)
        set_shape_change();

    // We assume that tensor ranks are static, thus shape_of doesn't need to update anything even if input shape is dynamic
    if (_node->is_type<shape_of>() && !input_shape_changed) {
        reset_shape_change();
        return;
    }

    // Do not update shapes in shape_of subraph if shape_of's input shape is not changed
    if (_node->is_in_shape_of_subgraph()) {
        bool subgraph_input_changed = false;
        for (size_t i = 0; i < dependant_shape_of_insts.size(); i++) {
            if (dependant_shape_of_insts[i]->shape_changed()) {
                subgraph_input_changed = true;
                break;
            }
        }
        if (!subgraph_input_changed) {
            GPU_DEBUG_TRACE_DETAIL << id() << ": skip shape_update, because it is in shape_of_subgrap and input shape is not changed\n";
            reset_shape_change();
            return;
        }
    }

    // Even though the predecessors' shapes are not changed, the output shape might be udpated by the mem_dep
    auto memory_deps = _node->get_const_memory_deps();
    for (auto& i : _node->get_shape_infer_dependencies()) {
        if (memory_deps.count(i) > 0) {
            continue;
        }
        if (i >= _deps.size())
            continue;

        if (_deps[i].first->get_node().is_in_shape_of_subgraph()) {
            bool can_skip = true;
            const auto& insts = _deps[i].first->dependant_shape_of_insts;
            for (auto& inst : insts) {
                can_skip &= !inst->shape_changed();
            }
            if (can_skip)
                continue;
        }

        input_shape_changed = true;
    }

    if (!input_shape_changed && !_node->generates_dynamic_output() && _impl_params->get_output_layout().is_static())
        return;

    std::vector<event::ptr> dependencies_events;
    auto queue_type = get_network().get_stream().get_queue_type();
    bool has_runtime_deps = false;
    for (auto& i : _node->get_shape_infer_dependencies()) {
        // Some primitives may have flexible count of deps (e.g. reshape), thus allow skipping some deps
        if (memory_deps.count(i) > 0 || i >= _node->get_dependencies().size()) {
            continue;
        }
        auto& dep = _node->get_dependency(i);
        auto dep_id = dep.id();
        // exclude fused node from memory_deps
        if (_node->is_fused_dep(i)) {
            break;
        }

        auto dep_mem = _network.get_output_memory(dep_id);
        memory_deps.insert({i, dep_mem});
        if (!get_node().is_type<shape_of>() && !dep.is_in_shape_of_subgraph()) {
            has_runtime_deps = true;

            // Events may be not created for in-order queue, so take them for OOO queue only
            if (_network.has_event(dep.id()) && queue_type == QueueTypes::out_of_order) {
                dependencies_events.push_back(_network.get_primitive_event(dep_id));
                GPU_DEBUG_TRACE_DETAIL << id() << ": shape infer waits for " << i << " dependency\n";
            }
        }
    }

    if (has_runtime_deps) {
        OV_ITT_SCOPED_TASK(ov::intel_gpu::itt::domains::intel_gpu_plugin, openvino::itt::handle("update_shape_sync: " + id()));
        if (!dependencies_events.empty() && queue_type == QueueTypes::out_of_order) {
            _network.get_stream().wait_for_events(dependencies_events);
        } else if (queue_type == QueueTypes::in_order) {
            _network.get_stream().finish();
        }
    }

    _impl_params->memory_deps = memory_deps;

    auto update_output_layout = [&](layout& layout, size_t idx) {
        auto data_padding = padding::max(_impl_params->get_output_layout(idx).data_padding, layout.data_padding);
        layout.data_padding = padding::max(_node->get_primitive()->get_output_padding(idx), data_padding);
        if (_impl_params->get_output_layout(idx) != layout) {
            GPU_DEBUG_TRACE_DETAIL << id() << ": update shape: was: " << _impl_params->get_output_layout(idx).to_short_string()
                                   << " now: " << layout.to_short_string() << std::endl;
            set_shape_change();
        }
        _impl_params->output_layouts[idx] = layout;
    };

    auto new_layouts = _node->type()->calc_output_layouts(*_node, *_impl_params);
    if (new_layouts.empty()) {
        auto new_layout = _node->type()->calc_output_layout(*_node, *_impl_params);
        update_output_layout(new_layout, 0);
    } else {
        for (size_t i = 0; i != new_layouts.size(); ++i) {
            auto new_layout = new_layouts[i];
            update_output_layout(new_layout, i);
        }
    }

    // Update descriptors of fused operations and set output_layout's shape to all fused ops
    // It's legal as long as fused ops don't change the shape
    for (auto& fused_prim : _impl_params->fused_desc) {
        fused_prim.output_layout.set_partial_shape(_impl_params->get_output_layout().get_partial_shape());
    }
}

event::ptr primitive_inst::realloc_if_needed() {
    OV_ITT_SCOPED_TASK(ov::intel_gpu::itt::domains::intel_gpu_plugin, openvino::itt::handle("realloc_if_needed: " + id()));
    GPU_DEBUG_GET_INSTANCE(debug_config);
    GPU_DEBUG_PROFILED_STAGE(instrumentation::pipeline_stage::memory_allocation);

    event::ptr ev = nullptr;
    if (_node->get_users().size() == 1 && _node->get_users().front()->is_type<concatenation>()) {
        auto concat_inst = _network.get_primitive(get_users().front()->id());
        if (concat_inst->can_be_optimized()) {
            if (!concat_inst->allocation_done_by_other) {
                concat_inst->realloc_if_needed();
                concat_inst->allocation_done_by_other = true;
            }
            this->_outputs[0] = concat_inst->_outputs[0];
            GPU_DEBUG_TRACE_DETAIL << id() << ": use concat user's memory " << this->_outputs[0]->buffer_ptr() << std::endl;
            return ev;
        }
    }
    // Update param if fake_alignment is available
    auto updated_params = _node->type()->get_fake_aligned_params(*_impl_params);
    auto actual_layout = updated_params.get_output_layout();
    OPENVINO_ASSERT(actual_layout.is_static(), "[GPU] Can't realloc mem for dynamic layout");

    // input_layout node is supposed to always use external memory in dynamic case
    if (_node->is_type<input_layout>())
        return ev;

    if (_node->is_type<assign>() || _node->is_type<read_value>()) {
        std::string variable_id = "";
        if (_node->is_type<assign>())
            variable_id = _node->as<assign>().get_primitive()->variable_id;
        else
            variable_id = _node->as<read_value>().get_primitive()->variable_id;
        get_network().update_variable_memory(variable_id, actual_layout);
        return ev;
    }

    bool can_reuse_buffer = _outputs[0] && actual_layout.count() <= max_output_layout_size;

    // Handle runtime dynamic concat optimization
    if (_node->is_type<concatenation>() && can_be_optimized() && allocation_done_by_other) {
        allocation_done_by_other = false;
        return ev;
    }

    auto current_shape = actual_layout.get_shape();
    auto& sp = *get_network().get_shape_predictor();
    auto dt_size = ov::element::Type(actual_layout.data_type).bitwidth();
    auto prealloc_info = sp.predict_preallocation_shape(id(), current_shape, dt_size, can_reuse_buffer);
    if (prealloc_info.first && sp.can_preallocate(ov::shape_size(prealloc_info.second) * dt_size)) {
        auto new_layout = actual_layout;
        new_layout.set_partial_shape(prealloc_info.second);
        updated_params.output_layouts[0] = new_layout;
    }

    if (can_reuse_buffer) {
        GPU_DEBUG_TRACE_DETAIL << id() << ": reuse previously allocated output buffer" << std::endl;
        if (_outputs[0]->get_layout() != actual_layout) {
            _outputs[0]->set_reused(true);
            _outputs[0] = _network.get_engine().reinterpret_buffer(*_outputs[0], actual_layout);
        }
        if (need_reset_output_memory() && !can_be_optimized()) {
            ev = _outputs[0]->fill(_network.get_stream());
        }
    } else {
        GPU_DEBUG_TRACE_DETAIL << id() << ": realloc output memory. "
                               <<  " Current buffer_size=" << max_output_layout_size
                               <<  " Requested buffer_size=" << actual_layout.count() << std::endl;
        _outputs = allocate_outputs(&updated_params, need_reset_output_memory(), true);
        // TODO : need to handle multiple outputs
        max_output_layout_size = updated_params.output_layouts[0].count();
    }
    _mem_allocated = true;
    // intermediate memory allocation is required for primitives consisting of multiple kernels in dynamic case
    {
        if (_impl == nullptr)
            return ev;
        const auto& ibuf_layouts = _impl->get_internal_buffer_layouts();
        if (ibuf_layouts.empty())
            return ev;

        for (size_t i = 0; i < ibuf_layouts.size(); ++i) {
            if (i < _intermediates_memory.size() && ibuf_layouts[i].bytes_count() <= max_intermediates_memory_sizes[i]) {
                // can reuse
                _intermediates_memory[i] = _network.get_engine().reinterpret_buffer(*_intermediates_memory[i], ibuf_layouts[i]);
            } else {
                // TODO: If there is a kernel which requires reset internal buffer in the future,
                // we'll need additional handle for that purpose like need_reset_output_memory
                bool need_reset = false;
                if (i < _intermediates_memory.size()) {
                    _intermediates_memory[i] = allocate_internal_buffer(i, need_reset);
                    max_intermediates_memory_sizes[i] = _intermediates_memory[i]->size();
                } else {
                    // i-th layout has not been allocated yet
                    _intermediates_memory.push_back(allocate_internal_buffer(i, need_reset));
                    max_intermediates_memory_sizes.push_back(_intermediates_memory[i]->size());
                }
            }
        }
    }
    return ev;
}

bool primitive_inst::use_async_compilation() {
    GPU_DEBUG_GET_INSTANCE(debug_config);
    GPU_DEBUG_IF(debug_config->disable_async_compilation) {
        return false;
    }
    return (_node->is_type<convolution>() ||
            _node->is_type<fully_connected>() ||
            _node->is_type<gemm>() ||
            _node->is_type<softmax>());
}

bool primitive_inst::update_impl() {
    OV_ITT_SCOPED_TASK(ov::intel_gpu::itt::domains::intel_gpu_plugin, openvino::itt::handle("update_impl: " + id()));
    GPU_DEBUG_PROFILED_STAGE(instrumentation::pipeline_stage::update_implementation);
    auto prev_impl_str =  _impl != nullptr ? _impl->get_kernel_name() : "nullptr";

    auto update_shape_info = [this, prev_impl_str](const kernel_impl_params& params) {
        mem_lock<int32_t> lock(_shape_info_memory, _network.get_stream());
        size_t offset = 0;
        for (size_t i = 0; i < _node->get_dependencies().size(); i++) {
            auto node_in_lay = _node->get_dependency(i).get_output_layout();
            if (node_in_lay.is_dynamic()) {
                auto pshape = params.get_input_layout(i).get_partial_shape();
                GPU_DEBUG_TRACE_DETAIL << id() << " : update shape_info for input[" << i << "]" << std::endl;
                auto input_shape_max_rank = layout::transform(pshape,
                                                              format::get_default_format(pshape.size()),
                                                              format::get_default_format(layout::max_rank())).to_shape();
                for (size_t j = 0; j < input_shape_max_rank.size(); ++j) {
                    GPU_DEBUG_TRACE_DETAIL << " shape_info[" << offset << "] = " << input_shape_max_rank[j] << std::endl;
                    lock[offset++] = static_cast<int32_t>(input_shape_max_rank[j]);
                }
                auto is_dynamic_pad = node_in_lay.data_padding.get_dynamic_pad_dims().sizes(format::get_default_format(layout::max_rank()));
                auto data_padding = params.input_layouts[i].data_padding;
                for (size_t j = 0; j < input_shape_max_rank.size(); ++j) {
                    if (is_dynamic_pad[j] == 1) {
                        auto lower_pads =
                            data_padding.lower_size().sizes(format::get_default_format(layout::max_rank()));
                        GPU_DEBUG_TRACE_DETAIL << " shape_info[" << offset << "] = " << lower_pads[j]
                                               << "(pad_before for input[" << i << "] " << j << "-th dim)" << std::endl;
                        lock[offset++] = lower_pads[j];  // pad_before
                        auto upper_pads =
                            data_padding.upper_size().sizes(format::get_default_format(layout::max_rank()));
                        GPU_DEBUG_TRACE_DETAIL << " shape_info[" << offset << "] = " << upper_pads[j]
                                               << "(pad_after for input[" << i << "] " << j << "-th dim)" << std::endl;
                        lock[offset++] = upper_pads[j];  // pad_after
                    }
                }
            }
        }
        for (size_t i = 0; i < _node->get_output_layouts().size(); i++) {
            auto node_out_lay = _node->get_output_layout(i);
            if (node_out_lay.is_dynamic()) {
                GPU_DEBUG_TRACE_DETAIL << id() << " : update shape_info for output[" << i << "]" << std::endl;
                auto pshape = params.get_output_layout(i).get_partial_shape();
                auto output_shape_max_rank = layout::transform(pshape,
                                                               format::get_default_format(pshape.size()),
                                                               format::get_default_format(layout::max_rank()))
                                                               .to_shape();
                for (size_t j = 0; j < output_shape_max_rank.size(); j++) {
                    GPU_DEBUG_TRACE_DETAIL << " shape_info[" << offset << "] = " << output_shape_max_rank[j] << std::endl;
                    lock[offset++] = static_cast<int32_t>(output_shape_max_rank[j]);
                }
                auto is_dynamic_pad = node_out_lay.data_padding.get_dynamic_pad_dims().sizes(format::get_default_format(layout::max_rank()));
                auto data_padding = params.output_layouts[i].data_padding;
                for (size_t j = 0; j < output_shape_max_rank.size(); j++) {
                    if (is_dynamic_pad[j] == 1) {
                        auto lower_pads = data_padding.lower_size().sizes(format::get_default_format(layout::max_rank()));
                        GPU_DEBUG_TRACE_DETAIL << " shape_info[" << offset << "] = " << lower_pads[j]
                                               << "(pad_before for output[" << i << "] " << j << "-th dim)" << std::endl;
                        lock[offset++] = lower_pads[j];
                        auto upper_pads = data_padding.upper_size().sizes(format::get_default_format(layout::max_rank()));
                        GPU_DEBUG_TRACE_DETAIL << " shape_info[" << offset << "] = " << upper_pads[j]
                                               << "(pad_after for output[" << i << "] " << j << "-th dim)" << std::endl;
                        lock[offset++] = upper_pads[j];  // pad_after
                    }
                }
            }
        }
        std::stringstream s;
        s << "shapes: ";
        for (size_t i = 0; i < offset; i++)
            s << lock[i] << " ";
        GPU_DEBUG_TRACE_DETAIL << id() << ": update dynamic impl " << prev_impl_str << " to new shape: " << s.str() << std::endl;
    };

    if (_impl != nullptr && (_impl->is_cpu() || can_be_optimized())) {
        // Return false if shape not changed, otherwise return true to trigger realloc_if_needed, but do not change impl itself
        return shape_changed();
    }

    if (!_node->is_type<data>() && !(_node->is_type<mutable_data>() && _node->get_dependencies().empty())) {
        // Update param if fake_alignment is available
        auto updated_params = _node->type()->get_fake_aligned_params(*_impl_params);
        // Change weights layout of `updated_params` to original one to have valid information
        // in _impl->_weights_reorder_params about required weights format after impl selection
        if (_node->is_type<fully_connected>() || _node->is_type<convolution>() || _node->is_type<deconvolution>()) {
            const auto weights_idx = _node->get_primitive()->input.size();
            const auto original_weights_memory = dep_memory_ptr(weights_idx);
            updated_params.weights_layout = optional_layout(original_weights_memory->get_layout());
        }

        auto updated_params_no_dyn_pad = updated_params;
        for (auto& i : updated_params_no_dyn_pad.input_layouts) {
            i.data_padding.set_dynamic_pad(tensor(0));
        }
        for (auto& o : updated_params_no_dyn_pad.output_layouts) {
            o.data_padding.set_dynamic_pad(tensor(0));
        }

        const auto& prog = get_network().get_program();
        auto& cache = prog->get_implementations_cache();
        std::shared_ptr<primitive_impl> cached_impl = nullptr;
        {
            cached_impl = cache.get(updated_params_no_dyn_pad);
            if (cached_impl) {
                _impl = cached_impl->clone();
                GPU_DEBUG_PROFILED_STAGE_CACHE_HIT(true);
                GPU_DEBUG_TRACE_DETAIL << id() << ": get impl from cache " << _impl->get_kernel_name() << std::endl;
            // impl is not replaced
            } else if (!shape_changed() && _impl != nullptr && _impl->is_dynamic()) {
                return false;
            }
        }
        if (!cached_impl) {
            if (_dynamic_impl) {
                if (use_async_compilation()) {
                    auto& compilation_context = prog->get_compilation_context();
                    compilation_context.push_task(updated_params_no_dyn_pad, [this, &compilation_context, updated_params_no_dyn_pad]() {
                        if (compilation_context.is_stopped())
                            return;
                        auto _program = get_network().get_program();
                        auto& cache = _program->get_implementations_cache();
                        {
                            // Check existense in the cache one more time as several iterations of model execution could happens and multiple compilation
                            // tasks created for same shapes
                            if (cache.has(updated_params_no_dyn_pad))
                                return;
                        }

                        if (!can_be_optimized()) {
                            auto impl = _node->type()->choose_impl(*_node, updated_params_no_dyn_pad);
                            if (impl->get_kernels_source().size() > 0) {
                                auto kernels = _program->get_kernels_cache().compile(updated_params_no_dyn_pad, impl->get_kernels_source());
                                impl->set_kernels(kernels);
                            }
                            cache.add(updated_params_no_dyn_pad, impl->clone());
                        }
                    });
                }
                if (!can_be_optimized())  {
                    _impl = _dynamic_impl->clone();
                    auto new_impl_params = _impl->canonicalize_shapes(*_impl_params);
                    _impl->update_dispatch_data(new_impl_params);
                    update_shape_info(new_impl_params);
                }
            } else {
                _impl = _node->type()->choose_impl(*_node, updated_params_no_dyn_pad);
                _impl->set_node_params(*_node);
                if (!can_be_optimized()) {
                    auto& kernels_cache = prog->get_kernels_cache();
                    auto kernels = kernels_cache.compile(updated_params_no_dyn_pad, _impl->get_kernels_source());
                    _impl->set_kernels(std::move(kernels));
                    cache.add(updated_params_no_dyn_pad, _impl->clone());
                }
                auto new_impl_str = _impl != nullptr ? _impl->get_kernel_name() : "nullptr";
                GPU_DEBUG_TRACE_DETAIL << id() << ": update impl from " << prev_impl_str << " to " << new_impl_str << std::endl;
            }
        }

        reset_shape_change();
    }
    // impl is replaced
    return true;
}

void primitive_inst::do_runtime_skip_reorder() {
    OV_ITT_SCOPED_TASK(ov::intel_gpu::itt::domains::intel_gpu_plugin, openvino::itt::handle("do_runtime_skip_reorder: " + id()));
    GPU_DEBUG_GET_INSTANCE(debug_config);
    GPU_DEBUG_IF(debug_config->disable_runtime_skip_reorder) {
        return;
    }
    if (can_be_optimized())
        return;

    if (_impl_params->fused_desc.size() > 0)
        return;

    // set successive reorder can_be_optimized if layouts are same
    for (auto u : get_user_insts()) {
        if (u->get_node().is_type<reorder>()) {
            if (is_input() && u->is_output())
                continue;
            // TODO: Skipped reorder + in_place concat is not supported yet. To support later.
            if (u->get_users().size() == 1 && u->get_users().front()->is_type<concatenation>() && u->get_users().front()->can_be_optimized())
                continue;
            auto out_port_idx = u->get_node().get_dependency_with_port(0).second;
            // If current node's output_node is not dynamic, the memory is already allocated at build time
            auto alloc_type = allocation_type::unknown;
            if (!get_node().is_dynamic_output_layout(out_port_idx) && static_cast<int64_t>(_outputs.size()) > out_port_idx) {
                alloc_type = _outputs[out_port_idx]->get_allocation_type();
            }
            if (alloc_type == allocation_type::usm_device && u->is_output())
                continue;
            GPU_DEBUG_TRACE_DETAIL << "[do runtime skip reorder] update shape for user " << u->id() << std::endl;
            u->update_shape();
            u->update_shape_done_by_other = true;

            if (u->_impl_params->get_input_layout() == u->_impl_params->get_output_layout()) {
                std::function<void(std::vector<std::shared_ptr<primitive_inst>>)> update_memory_dependencies;
                update_memory_dependencies = [&](std::vector<std::shared_ptr<primitive_inst>> users) {
                    for (auto& user : users) {
                        GPU_DEBUG_TRACE_DETAIL << "[do runtime skip reorder] add " << id() << " to restriction list of " << user->id() << std::endl;
                        user->_runtime_memory_dependencies.insert(id());
                        if (user->can_be_optimized())
                            update_memory_dependencies(user->get_user_insts());
                    }
                };

                update_memory_dependencies(u->get_user_insts());

                u->set_can_be_optimized(true);
                // Opt out reorder which has _needs_completion_event = true causes syncronization failed in dGPU.
                if (_needs_completion_event == false && u->_needs_completion_event == true) {
                    _needs_completion_event = true;
                }
                GPU_DEBUG_TRACE_DETAIL << "[do runtime skip reorder] set user " << u->id() << " as can_be_optimized" << std::endl;
            } else {
                GPU_DEBUG_TRACE_DETAIL << "[do runtime skip reorder] user " << u->id() << " cannot be optimized" << std::endl;
            }
        }
    }
}

void primitive_inst::do_runtime_in_place_concat() {
    OV_ITT_SCOPED_TASK(ov::intel_gpu::itt::domains::intel_gpu_plugin, openvino::itt::handle("do_runtime_in_place_concat: " + id()));
    GPU_DEBUG_GET_INSTANCE(debug_config);
    GPU_DEBUG_IF(debug_config->disable_runtime_buffer_fusing) {
        return;
    }
    if (update_shape_done_by_other) {
        return;
    }
    if (get_users().size() != 1) return;

    auto concat_inst = _network.get_primitive(get_users().front()->id());
    if (!concat_inst->get_node().is_type<concatenation>() || !concat_inst->get_node().can_be_optimized())
        return;
    // Currently does not support cascaded concats
    std::vector<std::shared_ptr<primitive_inst>> concat_preds;
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
    for (auto& pred : concat_inst->_deps) {
        pred_params.push_back(*pred.first->_impl_params);
        preds_layouts.push_back(pred.first->_impl_params->get_output_layout());
    }

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
            dep.first->set_shape_change();
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

bool primitive_inst::has_inner_networks() const {
    return (_impl_params->inner_nets.size() > 0);
}

event::ptr primitive_inst::execute(const std::vector<event::ptr>& events) {
    OV_ITT_SCOPED_TASK(ov::intel_gpu::itt::domains::intel_gpu_plugin, openvino::itt::handle("primitive_inst::execute: " + id()));
    const auto primitive_id = id();
    OPENVINO_ASSERT(_has_valid_input, primitive_id, " has invalid/unset input");
    GPU_DEBUG_GET_INSTANCE(debug_config);
    bool need_args_update = false;
    std::vector<event::ptr> dependencies;
    if (is_dynamic() && !has_inner_networks()) {
        do_runtime_in_place_concat();
        OPENVINO_ASSERT(_node != nullptr, "[GPU] Invalid primitive_inst object for dynamic shapes case: program_node can't be null");
        update_shape();

        // Check successor reorder if layouts are same
        // Need to set can_be_optimized for user reorder at predecessor because
        // if the user is can_be_optimized and output node then current nodes' output should be allocated to host.
        do_runtime_skip_reorder();
        if (_impl_params->output_layouts[0].count() == 0) {
            GPU_DEBUG_TRACE_DETAIL << id() << " : Skipping because output data is empty " << std::endl;
            auto ev = get_network().get_stream().create_user_event(true);
            update_shape_done_by_other = false; // reset
            return ev;
        }

        if (!is_valid_fusion()) {
            OV_ITT_SCOPED_TASK(ov::intel_gpu::itt::domains::intel_gpu_plugin, openvino::itt::handle("unfused_subgraph_exec: " + id()));
            auto subgraph = get_unfused_subgraph();

            for (auto& d : _deps) {
                if (!d.first->get_node().is_type<data>()) {
                    auto allocated_mem = d.first->output_memory_ptr();
                    auto actual_input_layout = d.first->get_output_layout();
                    auto& engine = _network.get_engine();
                    // Need to use actual layout, not the fake aligned memory layout
                    auto actual_mem = engine.reinterpret_buffer(*allocated_mem, actual_input_layout);
                    subgraph->set_input_data(d.first->id(), std::move(actual_mem));
                }
            }
            GPU_DEBUG_TRACE_DETAIL << "[Start] Executing unfused subgraph of " << id() << std::endl;
            auto outputs = subgraph->execute(events);
            GPU_DEBUG_TRACE_DETAIL << "[End] Finished executing unfused subgraph of " << id() << std::endl;

            auto last_fd = _impl_params->fused_desc.back();
            auto last_prim_id = last_fd.desc->id;

            OPENVINO_ASSERT(outputs.find(last_prim_id) != outputs.end(), "[GPU] Can't find output primitive ", last_prim_id, " for unfused subgraph");

            _outputs[0] = outputs.at(last_prim_id).get_memory();

            _impl_params->output_layouts[0] = subgraph->get_output_layout(last_prim_id);
            return outputs.at(last_prim_id).get_event();
        }

        // Try update impl if current impl is dynamic because opt kernel may be added to impl cache through async compilation.
        // Only try update weight and realloc when impl is updated.
        if (shape_changed() || !_impl || (!shape_changed() && _impl->is_dynamic())) {
            need_args_update = true;

            if (update_impl()) {
                auto ev = update_weights();
                if (ev)
                    dependencies.push_back(ev);
                auto ev_reset = realloc_if_needed();
                if (ev_reset)
                    dependencies.push_back(ev_reset);
            }
        }

        OPENVINO_ASSERT(_impl_params->get_output_layout().is_static(),
                        "[GPU] Can't execute ", primitive_id, " primitive as output layout is dynamic in runtime");
    }
    update_shape_done_by_other = false; // reset
    OPENVINO_ASSERT(_impl != nullptr, "[GPU] Implementation is nullptr for ", primitive_id,  " primitive");

    // Output buffer may be changed under the following conditions, so we need to set args to kernel on each iteration
    if ((is_dynamic() && need_args_update) || has_mutable_input() || is_output()) {
        set_arguments();
    }
    on_execute();

    GPU_DEBUG_TRACE << id() << ": execute " << _impl->get_kernel_name() << " (is_dynamic=" << _impl->is_dynamic() << ", "
                    << "can_be_optimized=" << can_be_optimized() << ")" << std::endl;

    const bool out_of_order_queue = get_network().get_stream().get_queue_type() == QueueTypes::out_of_order;
    if (_exec_deps.empty() && dependencies.empty()) {
        dependencies = events;
    } else {
        // Prepare dependencies events in case of OOO queue, CPU implementation,
        // or optimized_out impl which has CPU users (needs_completion_event() && !is_output() condition)
        if (out_of_order_queue || _impl->is_cpu() || (can_be_optimized() && needs_completion_event() && !is_output())) {
            dependencies.reserve(dependencies.size() + _exec_deps.size());
            for (auto& input : _exec_deps) {
                if (input->is_input() && !out_of_order_queue)
                    continue;
                auto id = input->id();
                try {
                    // if the requested event does not exists it means that it has not been executed, so the processing_order is
                    // wrong or synchronization failed.
                    auto ev = get_network().get_primitive_event(id);
                    dependencies.emplace_back(ev);
                } catch (const std::out_of_range& oor) {
                    OPENVINO_ASSERT(false, "[GPU] execution order corrupted: ", oor.what());
                }
            }
        }
    }

    // Replace multiple events with single grouped event in case of barriers synchronization to prevent `_last_barrier_ev` usage as a dependency
    // event of optimized_out instance's users, which may lead to unwanted extra synchronization of CPU impls with GPU kernels
    if (_node && _node->is_in_shape_of_subgraph() && can_be_optimized() && dependencies.size() > 1 && out_of_order_queue) {
        auto grouped_ev = get_network().get_stream().group_events(dependencies);
        dependencies = {grouped_ev};
    }

    {
        GPU_DEBUG_PROFILED_STAGE(instrumentation::pipeline_stage::inference);
        auto ev = _impl->execute(dependencies, *this);

        GPU_DEBUG_IF(!debug_config->dump_profiling_data.empty()) {
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

        return ev;
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
            dependant_shape_of_insts.push_back(_network.get_primitive(shape_of->id()));
        }
    }
}

void primitive_inst::rebuild_deps(
    std::unordered_map<primitive_id, std::shared_ptr<primitive_inst>> const& primitives) {

    _deps.resize(_dep_ids.size());
    for (size_t i = 0; i < _dep_ids.size(); i++) {
        OPENVINO_ASSERT((primitives.count(_dep_ids[i].first) > 0),
                        _dep_ids[i].first, "is not found in primitives while rebuilding _deps");
        _deps[i] = {primitives.at(_dep_ids[i].first), _dep_ids[i].second};
    }
}

void primitive_inst::rebuild_exec_deps(
    std::unordered_map<primitive_id, std::shared_ptr<primitive_inst>> const& primitives) {

    _exec_deps.resize(_exec_dep_ids.size());
    for (size_t i = 0; i < _exec_dep_ids.size(); i++) {
        OPENVINO_ASSERT((primitives.count(_exec_dep_ids[i]) > 0),
                        _exec_dep_ids[i], "is not found in primitives while rebuilding _exec_deps");
        _exec_deps[i] = primitives.at(_exec_dep_ids[i]);
    }
}

primitive_inst::primitive_inst(network& network)
    : _network(network)
    , _node(nullptr)
    , _impl_params(make_unique<kernel_impl_params>())
    , _impl(nullptr)
    , _dynamic_impl(nullptr)
    , _outputs({memory::ptr()})
    , _reordered_weights_cache(network.get_weights_cache_capacity())
    , _output_changed(false)
    , _mem_allocated(false)
    , _type(nullptr) {}

primitive_inst::primitive_inst(network& network, program_node const& node, bool allocate_memory)
    : _network(network)
    , _node(&node)
    , _node_output_layout(node.get_output_layout())
    , _impl_params(node.get_kernel_impl_params())
    , _impl(node.get_selected_impl() ? node.get_selected_impl()->clone() : nullptr)
    , _dynamic_impl(nullptr)
    , _runtime_memory_dependencies(node.get_memory_dependencies())
    , _outputs({memory::ptr()})
    , _reordered_weights_cache(network.get_weights_cache_capacity())
    , _output_changed(false)
    , _is_dynamic(node.is_dynamic() || node.generates_dynamic_output())
    , _type(node.type())
    , _id(node.id())
    , _org_id(node.get_org_primitive_id())
    , _is_input(node.is_input())
    , _is_output(node.is_output())
    , _inputs_memory_count(node.get_inputs_count())
    , _outputs_memory_count(node.get_outputs_count())
    , _fused_mem_count(node.get_fused_inputs_count())
    , _fused_mem_offset((_fused_mem_count > 0 && node.has_fused_dep()) ? node.get_first_fused_dep_idx() : 0)
    , _can_be_optimized(node.can_be_optimized())
    , _can_share_buffer(node.can_share_buffer())
    , _is_constant(node.is_constant())
    , _needs_completion_event(is_any_user_cpu(node.get_users()) || node.is_output()) {
    // When dynamic shape node has huge upper boundary which causes bigger mem size than system max allocable mem size, do not allocate in build time.
    auto output_layout = node.get_output_layout();
    if (allocate_memory && node.is_dynamic() && (!network.get_engine().check_allocatable(output_layout, allocation_type::usm_host))) {
        allocate_memory = false;
    }
    _mem_allocated = allocate_memory;
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
    if (_impl) {
        _impl->set_node_params(node);
        if (_impl->is_dynamic() && !_impl->is_cpu()) {
            GPU_DEBUG_TRACE_DETAIL << id() << ": initialize impl with dynamic impl " << _impl->get_kernel_name() << std::endl;
            _dynamic_impl = _impl->clone();
            // Actual shape info layout is the following:
            // input_0 -> input_1, ..., fused_dep_0, fused_dep1, ..., output_0, output_1, ...
            // For each tensor we save max_rank dimensions in [bfvuwzyx] order
            size_t num_dynamic_pads = 0;
            for (auto& in : _node->get_dependencies()) {
                const auto& dyn_pad_dims = in.first->get_output_layout(false).data_padding.get_dynamic_pad_dims().sizes();
                num_dynamic_pads += std::accumulate(dyn_pad_dims.begin(), dyn_pad_dims.end(), static_cast<int32_t>(0));
            }
            for (auto& o : _node->get_output_layouts()) {
                const auto& dyn_pad_dims = o.data_padding.get_dynamic_pad_dims().sizes();
                num_dynamic_pads += std::accumulate(dyn_pad_dims.begin(), dyn_pad_dims.end(), static_cast<int32_t>(0));
            }
            const int64_t buffers_count = _node->get_dependencies().size() + _node->get_outputs_count();
            const int64_t shape_elements = buffers_count * layout::max_rank() + num_dynamic_pads * 2 /*pad_before + pad_after*/;
            _shape_info_memory = _network.get_engine().allocate_memory(layout{{shape_elements}, data_types::i32, format::bfyx});
        }
    }
    _impl_params->strm = _network.get_stream_ptr();
    if (_outputs[0])
        max_output_layout_size = _outputs[0]->get_layout().get_tensor().count();
}

memory::ptr primitive_inst::allocate_internal_buffer(size_t idx, bool reset) {
    if (_impl == nullptr || _outputs.empty() || _outputs[0] == nullptr)
        return nullptr;
    const auto& ibuf_layouts = _impl->get_internal_buffer_layouts();
    if (ibuf_layouts.empty())
        return nullptr;

    auto device_mem_acc = [&](size_t a, std::pair<std::shared_ptr<primitive_inst>, int32_t> b) {
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
    // check if there is any device mem input
    if (engine.supports_allocation(allocation_type::usm_device)) {
        for (const auto& dep : inst_deps) {
            if (!dep.first->mem_allocated()) continue;
            if (dep.first->output_memory().get_allocation_type() == allocation_type::usm_device) {
                input_device_mem = true;
                break;
            }
        }
    }
    // allocate intermediate memory for the updated layout of buffer
    auto layout = ibuf_layouts[idx];
    GPU_DEBUG_LOG << "[" << _node->id() << ": internal buf " << idx << "]" << std::endl;
    auto alloc_type = allocation_type::unknown;
    if ((int64_t)available_device_mem_size - (int64_t)layout.bytes_count() >= 0 &&
        (input_device_mem || _node->get_preferred_impl_type() == impl_types::onednn)) {
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
    const auto& ibuf_layouts = _impl->get_internal_buffer_layouts();
    if (ibuf_layouts.empty())
        return;

    // allocate intermediate memory for the updated layout of buffer
    std::vector<memory::ptr> intermediates_memory;
    for (size_t i = 0; i < ibuf_layouts.size(); ++i) {
        if (ibuf_layouts[i].get_linear_size() == 0)
            continue;
        intermediates_memory.push_back(allocate_internal_buffer(i, reset));
        max_intermediates_memory_sizes.push_back(intermediates_memory[i]->size());
    }
    _intermediates_memory = intermediates_memory;
}

event::ptr primitive_inst::update_weights() {
    OV_ITT_SCOPED_TASK(ov::intel_gpu::itt::domains::intel_gpu_plugin, openvino::itt::handle("update_weights: " + id()));
    GPU_DEBUG_PROFILED_STAGE(instrumentation::pipeline_stage::update_weights);
    if (!_impl)
        return nullptr;

    bool weightable_node = _node->is_type<fully_connected>() || _node->is_type<convolution>() || _node->is_type<deconvolution>();
    if (!weightable_node)
        return nullptr;

    auto& engine = _network.get_engine();
    auto reorder_kernel_params = _impl->get_weights_reorder_kernel_params();

    if (reorder_kernel_params)
        reorder_kernel_params->prog = get_network().get_program().get();

    auto weights_idx = _node->get_primitive()->input.size();
    auto original_weights_memory = dep_memory_ptr(weights_idx);
    auto original_layout = original_weights_memory->get_layout();

    if (!reorder_kernel_params) {
        // If kernel doesn't says that it doesn't require weights reorder, but weights were reordered previously, then
        // incorrect memory buffer may be assigned, so reset cached weights for such case
        _reordered_weights_cache.add(original_layout, original_weights_memory);
        _impl_params->weights_layout = optional_layout(original_layout);
    } else {
        auto expected_layout = reorder_kernel_params->get_output_layout();
        // Set original partial shape, because it may be lost during kernel_selector::weights_tensor -> layout conversion
        expected_layout.set_partial_shape(original_layout.get_partial_shape());
        _impl_params->weights_layout = optional_layout(expected_layout);

        if (_reordered_weights_cache.has(expected_layout)) {
            GPU_DEBUG_PROFILED_STAGE_CACHE_HIT(true);
            GPU_DEBUG_TRACE_DETAIL << id() << ": reuse weights for " << expected_layout.to_short_string() << std::endl;
            return nullptr;
        } else if (original_layout.compatible(expected_layout)) {
            GPU_DEBUG_PROFILED_STAGE_CACHE_HIT(true);
            GPU_DEBUG_TRACE_DETAIL << id() << ": reinterpret original weights memory from " << original_layout.to_short_string()
                                           << " to " << expected_layout.to_short_string() << std::endl;
            _reordered_weights_cache.add(expected_layout, engine.reinterpret_buffer(*original_weights_memory, expected_layout));
            return nullptr;
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

                auto factory = WeightsReordersFactory::get(impl_types::ocl, shape_types::static_shape);
                auto reorder_impl = factory(*reorder_kernel_params);
                auto& kernels_cache = get_network().get_program()->get_kernels_cache();
                auto kernels = kernels_cache.compile(*reorder_kernel_params, reorder_impl->get_kernels_source());
                OPENVINO_ASSERT(kernels.size() == 1, "[GPU] Expected number of compiled kernels is 1, but got ", kernels.size());
                reorder_impl->set_kernels(kernels);

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
            auto ev = reorder_impl->execute({}, *reorder_inst);

            GPU_DEBUG_GET_INSTANCE(debug_config);
            GPU_DEBUG_IF(!debug_config->dump_profiling_data.empty()) {
                stream.wait_for_events({ev});
            }

            return ev;
        }
    }

    GPU_DEBUG_PROFILED_STAGE_CACHE_HIT(true);

    return nullptr;
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
                                            const std::set<primitive_id>& memory_dependencies,
                                            uint32_t net_id,
                                            bool is_internal,
                                            size_t idx,
                                            bool reset,
                                            bool is_output_buffer,
                                            memory* curr_memory,
                                            bool runtime_alloc) {
    auto layout = impl_params.get_output_layout(idx);
    OPENVINO_ASSERT(layout.is_static() || layout.has_upper_bound(), "[GPU] Can't allocate output for dynamic layout");
    auto device_mem_acc = [&](size_t a, const cldnn::layout& l) {
        // Input shape may be dynamic is some cases (shape_of). It means that output shape of node doesn't depend on input shape
        // and out memory can be allocated on program build stage.
        if (l.is_static())
            return a + l.bytes_count();

        return a;
    };

    layout = cldnn::layout(layout.get_partial_shape().get_max_shape(), layout.data_type, layout.format, layout.data_padding);
    bool usm_device_allocatable = true;
    const auto& total_device_input_mem_size = std::accumulate(impl_params.input_layouts.begin(), impl_params.input_layouts.end(), (uint64_t)0, device_mem_acc);
    if (total_device_input_mem_size > _engine.get_device_info().max_global_mem_size)
        usm_device_allocatable = false;

    bool reusable_across_network = (runtime_alloc && _node.is_dynamic_output_layout()) ? !reset : !user_requesting_mem_reuse_false(_node);

    // Do not use memory pool for nodes from shape_of subgraphs, because such nodes mostly use CPU impls and may be executed in parallel with predecessors
    // GPU kernels and cause accuracy problems. This significantly improves performance (because provides an ability not to synchronize shape_of subgraphs
    // execution with other nodes) at the cost of tiny increase in memory consumption.
    if (_node.is_in_shape_of_subgraph())
        reusable_across_network = false;

    // For outputs, cpu prim we want to have lockable alloc type
    // Also if the successor of a node is an cpu, then memory needs to be lockable.
    bool is_cpu = _node.get_selected_impl() ? _node.get_selected_impl()->is_cpu() : false;
    auto use_lockable_memory =
        is_output_buffer || is_cpu ||
        has_any_cpu_user_not_shape_of(_node.get_users()) ||
        !_engine.supports_allocation(allocation_type::usm_device) ||
        (_node.is_shape_infer_dep() && _engine.get_device_info().dev_type == device_type::integrated_gpu);
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
    } else if (!_node.can_share_buffer() || _node.can_be_optimized() || _node.is_output()) {
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
    for (size_t i = 0; i < get_node().get_outputs_count() ; ++i) {
        auto impl_params = updated_params != nullptr ? *updated_params : *_impl_params;
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
    return outputs;
}

std::vector<std::shared_ptr<primitive_inst>> primitive_inst::build_exec_deps(
    std::vector<std::pair<std::shared_ptr<primitive_inst>, int32_t>> const& deps) {
    std::vector<std::shared_ptr<primitive_inst>> exec_deps;
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
        for (auto& dep : _node->get_dependencies()) {
            if (dep.first->is_type<data>()) {
                auto& data_node = dep.first->as<data>();
                auto data_prim = *data_node.get_primitive();
                // mem field of original primitive can be nullified during transfer_memory_to_device pass, thus use mem from program_node
                data_prim.mem = data_node.get_attached_memory_ptr();
                t.add(data_prim);
            } else {
                input_layout in_prim(dep.first->id(), dep.first->get_output_layout());
                t.add(in_prim);
            }
            outer_dep_ids.push_back(dep.first->id());
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
                if (fd.has_outer_dep()) {
                    if (std::find_if(outer_dep_ids.begin(), outer_dep_ids.end(), [&](const primitive_id& pid) {
                            return pid == in.pid;
                        }) == outer_dep_ids.end()) {
                        size_t dep_id = fd.outer_dep_start_idx;
                        in = _node->get_dependency(dep_id).id();
                    }
                }
            }
            t.add_primitive(prim);
            outer_dep_ids.push_back(prim->id);
        }
        // Samely, need to update dependency of the current fused nodes' input primitive ids with those in the current program
        auto prim_of_fused_node = std::const_pointer_cast<primitive>(_impl_params->desc);
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
            ov::intel_gpu::allow_new_shape_infer(true)
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

    auto fuse_descriptors = _impl_params->fused_desc;
    if (fuse_descriptors.empty())
        return true;
    std::vector<fused_primitive_desc> fused_eltwise_prims;
    for (auto& fd : fuse_descriptors) {
        if (fd.is_type<eltwise>() || fd.is_type<activation>()) {
            fused_eltwise_prims.push_back(fd);
        } else {
            OPENVINO_ASSERT("[GPU] Unsupported fused operation in dynamic shape : ", fd.desc->id);
        }
    }

    if (fused_eltwise_prims.empty())
        return true;

    auto out_pshape = _impl_params->get_output_layout().get_partial_shape();
    for (auto& fd : fused_eltwise_prims) {
        auto outer_dep_idx = fd.outer_dep_start_idx;
        if (outer_dep_idx < 0) // no outer dep
            continue;
        OPENVINO_ASSERT(fd.total_num_deps == 2, "[GPU] Unexpected count of dependencies in dynamic fusion for eltwise or activation");
        OPENVINO_ASSERT(outer_dep_idx < 0 || static_cast<int32_t>(_deps.size()) > outer_dep_idx, "[GPU] Invalid fused dependency idx");
        auto outer_dep = _deps[outer_dep_idx];

        auto outer_dep_pshape = outer_dep.first->_impl_params->get_output_layout().get_partial_shape();
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
            auto gemm_layout = _impl_params->get_output_layout();
            auto data_layout = outer_dep.first->_impl_params->get_output_layout();
            auto gemm_dims = onednn::convert_gemm_tensor(gemm_layout.get_tensor(),
                                                         cldnn::format::dimension(gemm_layout.format),
                                                         false);

            auto data_dims = onednn::convert_gemm_tensor(data_layout.get_tensor(),
                                                         cldnn::format::dimension(data_layout.format),
                                                         false);

            if (gemm_dims[0] != data_dims[0])
                return false;
        }
#endif

        // We check that broadcasting of extra input is possible and it doesn't change output shape. If it output shape is changed, then
        // some dimension of dep_pshape is greater than out_pshape
        if (!can_broadcast || merged_shape != out_pshape)
            return false;
    }

    return true;
}

void primitive_inst::add_profiling_data(instrumentation::pipeline_stage stage, bool cache_hit, int64_t time, bool per_iter_mode) {
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
            cache_hit
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

static primitive_id find_dep_by_mem(const cldnn::primitive_inst* p_inst, memory& mem_ptr, int max_dist = 5) {
    std::vector<std::pair<primitive_id, int>> queue;
    size_t head = 0;

    for (auto& p_inst : p_inst->dependencies())
        queue.emplace_back(std::make_pair(p_inst.first->id(), 0));

    const network& const_network = p_inst->get_network();
    while (head < queue.size()) {
        auto curr_item = queue.at(head);
        auto curr_prim = const_network.get_primitive(curr_item.first);

        if (p_inst->get_network().get_engine().is_the_same_buffer(mem_ptr, curr_prim->output_memory()))
            return curr_prim->id();

        if (max_dist > curr_item.second)
            for (auto& p_inst : curr_prim->dependencies())
                queue.emplace_back(std::make_pair(p_inst.first->id(), curr_item.second+1));

        head += 1;
    }

    return "NOT_FOUND";
}

// Cache blob format:
//     [ kernel_impl_params ]
//     [ primitive_impl ]
//     [ member variables of primitive_inst ]
//     [ output memory information ]
//     [ memory dependency information ]
//     [ execution dependency information ]
//     [ intermediate memory information ]
void primitive_inst::save(cldnn::BinaryOutputBuffer& ob) const {
    _impl_params->save(ob);
    ob.setKernelImplParams(_impl_params.get());

    ob << _node_output_layout;
    ob << has_mutable_input();
    ob << mem_allocated();
    ob << is_dynamic();
    ob << _node->get_primitive()->type_string();
    ob << id();
    ob << org_id();
    ob << is_input();
    ob << is_output();
    ob << inputs_memory_count();
    ob << outputs_memory_count();
    ob << get_fused_mem_count();
    ob << get_fused_mem_offset();
    ob << can_be_optimized();
    ob << can_share_buffer();
    ob << is_constant();
    ob << needs_completion_event();

    if (type() == cldnn::data::type_id()) {
        return;
    }

    ob << _outputs.size();
    for (size_t i = 0; i < _outputs.size(); ++i) {
        if (_outputs[i] == nullptr) {
            ob << true;
        } else {
            ob << false;
            ob << _outputs[i]->get_layout();
            const auto _allocation_type = _outputs[i]->get_allocation_type();
            ob << make_data(&_allocation_type, sizeof(_allocation_type));
        }
    }

    bool can_reuse_memory = true;
    if (user_requesting_mem_reuse_false(*_node)) {
        can_reuse_memory = false;
    }
    ob << can_reuse_memory;

    ob << _node->get_memory_dependencies();

    ob << _deps.size();
    for (const auto& dep : _deps) {
        ob << dep.first->id();
        ob << dep.second;
    }

    ob << _exec_deps.size();
    for (const auto& dep : _exec_deps) {
        ob << dep->id();
    }

    for (size_t i = 0; i < _outputs.size(); ++i) {
        if (_outputs[i] != nullptr) {
            if (!mem_allocated())
                ob << find_dep_by_mem(this, output_memory(i));
        }
    }

    ob << _intermediates_memory.size();
    for (const auto& ibuf : _intermediates_memory) {
        ob << ibuf->get_layout();
        const auto _allocation_type = ibuf->get_allocation_type();
        ob << make_data(&_allocation_type, sizeof(_allocation_type));
    }

    if (_impl != nullptr) {
        ob << true;
        _impl->set_cached_kernel_ids(_network.get_program()->get_kernels_cache());
        ob << _impl;
    } else {
        ob << false;
    }
}

int32_t primitive_inst::get_index_in_deps(memory::cptr arg) const {
    for (uint32_t idx = 0; idx < _deps.size(); ++idx) {
        if (arg == dep_memory_ptr(idx))
            return idx;
    }

    OPENVINO_THROW("[get_index_in_deps]: not found in _deps");
}

void primitive_inst::load(cldnn::BinaryInputBuffer& ib) {
    _impl_params->load(ib);
    ib.setKernelImplParams(_impl_params.get());

    ib >> _node_output_layout;
    ib >> _has_mutable_input;
    ib >> _mem_allocated;
    ib >> _is_dynamic;
    std::string type_str;
    ib >> type_str;
    _type = cldnn::prim_map_storage::instance().get_type_id(type_str);
    ib >> _id;
    ib >> _org_id;
    ib >> _is_input;
    ib >> _is_output;
    ib >> _inputs_memory_count;
    ib >> _outputs_memory_count;
    ib >> _fused_mem_count;
    ib >> _fused_mem_offset;
    ib >> _can_be_optimized;
    ib >> _can_share_buffer;
    ib >> _is_constant;
    ib >> _needs_completion_event;

    if (type() == cldnn::data::type_id()) {
        return;
    }

    // mem_allocated : it is true if the output memory is allocated by this layer, and
    //                 false if this layer reuses output memory that is allocated by other layer.
    // is_output_null : it is true if the output memory is not allocated yet and false otherwise.
    size_t num_outputs;
    std::vector<bool> is_output_null;
    std::vector<layout> output_layouts;
    std::vector<allocation_type> allocation_types;

    ib >> num_outputs;
    is_output_null.resize(num_outputs);
    for (size_t i = 0; i < num_outputs; ++i) {
        bool is_null;
        ib >> is_null;
        is_output_null[i] = is_null;
        if (!is_null) {
            layout output_layout = layout();
            ib >> output_layout;
            output_layouts.emplace_back(output_layout);

            allocation_type _allocation_type = allocation_type::unknown;
            ib >> make_data(&_allocation_type, sizeof(_allocation_type));
            allocation_types.emplace_back(_allocation_type);
        }
    }

    bool can_reuse_memory;
    ib >> can_reuse_memory;

    std::set<primitive_id> _node_mem_deps;
    ib >> _node_mem_deps;
    _runtime_memory_dependencies = _node_mem_deps;

    size_t vector_size = 0UL;
    ib >> vector_size;
    for (size_t i = 0; i < vector_size; ++i) {
        primitive_id dep_id;
        int32_t dep_idx;
        ib >> dep_id >> dep_idx;
        _dep_ids.emplace_back(std::pair<primitive_id, int32_t>(dep_id, dep_idx));
    }

    ib >> vector_size;
    _exec_dep_ids.resize(vector_size);
    for (auto& el : _exec_dep_ids) {
        ib >> el;
    }

    _outputs.resize(num_outputs);
    for (size_t i = 0; i < num_outputs; ++i) {
        _outputs[i] = nullptr;
        if (!is_output_null[i]) {
            if (!_mem_allocated) {
                std::string dep_id;
                ib >> dep_id;
                if (dep_id.compare("NOT_FOUND") != 0 && get_network().get_primitive(dep_id)->output_memory_ptr() != nullptr) {
                    _outputs[i] = get_network().get_engine().reinterpret_buffer(get_network().get_primitive(dep_id)->output_memory(), output_layouts[i]);
                } else if (type() == cldnn::mutable_data::type_id()) {
                    _outputs[i] = get_network().get_engine().allocate_memory(output_layouts[i], allocation_types[i]);
                }
            } else {
                if ((!can_share_buffer()) || can_be_optimized() || is_output()) {
                    _outputs[i] = get_network().get_engine().allocate_memory(output_layouts[i], allocation_types[i]);
                } else {
                    _outputs[i] = get_network().get_memory_pool().get_memory(output_layouts[i], id(), get_network_id(), _node_mem_deps,
                                                                            allocation_types[i], can_reuse_memory);
                }
            }
        }
    }
    _output_changed = false;

    ib >> vector_size;
    _intermediates_memory.resize(vector_size);
    for (size_t i = 0; i < vector_size; i++) {
        layout ibuf_layout = layout();
        ib >> ibuf_layout;
        allocation_type _allocation_type;
        ib >> make_data(&_allocation_type, sizeof(_allocation_type));

        _intermediates_memory[i] = get_network().get_memory_pool().get_memory(ibuf_layout, id(), get_network_id(),
                                                                            _node_mem_deps, _allocation_type, true, true);
    }

    bool has_impl;
    ib >> has_impl;
    if (has_impl) {
        _impl.reset();
        ib >> _impl;
    }
}

}  // namespace cldnn
