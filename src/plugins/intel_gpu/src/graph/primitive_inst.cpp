// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "primitive_inst.h"
#include "data_inst.h"
#include "mutable_data_inst.h"
#include "generic_layer_inst.h"
#include "input_layout_inst.h"
#include "arg_max_min_inst.h"
#include "fully_connected_inst.h"
#include "convolution_inst.h"
#include "crop_inst.h"
#include "deconvolution_inst.h"
#include "shape_of_inst.h"
#include "gemm_inst.h"
#include "experimental_detectron_roi_feature_extractor_inst.hpp"
#include "compilation_context.hpp"

#include "intel_gpu/plugin/common_utils.hpp"
#include "intel_gpu/graph/network.hpp"
#include "intel_gpu/graph/serialization/set_serializer.hpp"
#include "intel_gpu/runtime/engine.hpp"
#include "intel_gpu/runtime/memory.hpp"
#include "intel_gpu/runtime/error_handler.hpp"
#include "intel_gpu/runtime/debug_configuration.hpp"

#include "json_object.h"
#include <string>
#include <stack>
#include <vector>
#include <memory>
#include <algorithm>

namespace cldnn {
namespace {

bool is_optimized_output_user(const program_node* user) {
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

bool is_output_buffer(const program_node& node) {
    if (node.is_output())
        return true;

    // Try to recursively find any optimized out user which is also network output
    for (const auto& user : node.get_users()) {
        if (is_optimized_output_user(user)) {
            return true;
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
}  // namespace

bool is_any_user_cpu(const std::list<const program_node*>& users) {
    for (const auto& user : users) {
        if (is_user_cpu(user))
            return true;
    }
    return false;
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
                        " while memory object was allocated for ", &mem_engine, "(",
                        mem_engine.get_device_info().dev_name, ")");

        switch (params.mem_type) {
        case shared_mem_type::shared_mem_vasurface:
        case shared_mem_type::shared_mem_image:
            if (!layout.format.is_image_2d())
                CLDNN_ERROR_MESSAGE(_node->id(), "Attempt to set user-supplied input or output image instead of a buffer");
            break;
        case shared_mem_type::shared_mem_buffer:
        case shared_mem_type::shared_mem_dxbuffer:
            if (layout.format.is_image_2d())
                CLDNN_ERROR_MESSAGE(_node->id(), "Attempt to set user-supplied input or output buffer instead of an image");
            break;
        case shared_mem_type::shared_mem_usm:
            break;
        default:
            CLDNN_ERROR_MESSAGE(_node->id(), "Attempt to set user-supplied input or output memory of unknown/invalid type");
            break;
        }
    }
}

void primitive_inst::set_output_memory(memory::ptr mem_new, bool check, size_t idx) {
    auto& eng = _network.get_engine();
    // skip all the buzz if no action actually required
    if (_outputs[idx] && eng.is_the_same_buffer(*mem_new, *_outputs[idx])) {
        return;
    }

    auto ol = _impl_params->get_output_layout(idx);

    if (check)
        check_memory_to_set(*mem_new, ol);

    if (is_constant()) {
        mem_new->copy_from(_network.get_stream(), *_outputs[idx]);
    } else {
        _outputs[idx] = mem_new;
    }
}

void primitive_inst::update_shape() {
    GPU_DEBUG_PROFILED_STAGE(instrumentation::pipeline_stage::shape_inference);

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

    // Even though the predecessors' shapes are not changed, the output shape might be udpated by the mem_dep
    auto memory_deps = _node->get_const_memory_deps();
    for (auto& i : _node->get_shape_infer_dependencies()) {
        if (memory_deps.count(i) > 0) {
            continue;
        }
        input_shape_changed = true;
    }

    // We assume that tensor ranks are static, thus shape_of doesn't need to update anything even if input shape is dynamic
    if (_node->is_type<shape_of>() && !input_shape_changed)
        return;

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
        // Events may be not created for in-order queue, so take them for OOO queue only
        if (_network.has_event(dep.id()) && queue_type == QueueTypes::out_of_order) {
            dependencies_events.push_back(_network.get_primitive_event(dep_id));
            GPU_DEBUG_TRACE_DETAIL << id() << ": shape infer waits for " << i << " dependency\n";
        }
        auto dep_mem = _network.get_output_memory(dep_id);
        memory_deps.insert({i, dep_mem});
        has_runtime_deps = true;
    }

    if (has_runtime_deps) {
        if (!dependencies_events.empty() && queue_type == QueueTypes::out_of_order) {
            _network.get_stream().wait_for_events(dependencies_events);
        } else if (queue_type == QueueTypes::in_order) {
            _network.get_stream().finish();
        }
    }

    _impl_params->memory_deps = memory_deps;

    auto update_output_layout = [&](layout& layout, size_t idx) {
        layout.data_padding = padding::max(_node->get_primitive()->output_paddings[idx], layout.data_padding);
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

void primitive_inst::realloc_if_needed() {
    GPU_DEBUG_GET_INSTANCE(debug_config);
    GPU_DEBUG_PROFILED_STAGE(instrumentation::pipeline_stage::memory_allocation);


    // Update param if fake_alignment is available
    auto updated_params = _node->type()->get_fake_aligned_params(*_impl_params);
    auto actual_layout = updated_params.get_output_layout();
    OPENVINO_ASSERT(actual_layout.is_static(), "[GPU] Can't realloc mem for dynamic layout");

    // input_layout node is supposed to always use external memory in dynamic case
    if (_node->is_type<input_layout>())
        return;

    bool can_reuse_buffer = _outputs[0] && actual_layout.count() <= max_output_layout_size;

    if (can_reuse_buffer) {
        GPU_DEBUG_TRACE_DETAIL << id() << ": reuse previously allocated output buffer" << std::endl;
        _outputs[0] = _network.get_engine().reinterpret_buffer(*_outputs[0], actual_layout);
    } else {
        GPU_DEBUG_TRACE_DETAIL << id() << ": realloc output memory. "
                               <<  " Current buffer_size=" << max_output_layout_size
                               <<  " Requested buffer_size=" << actual_layout.count() << std::endl;
        _outputs = allocate_outputs(&updated_params);
        // TODO : need to handle multiple outputs
        max_output_layout_size = updated_params.output_layouts[0].count();
    }
    // intermediate memory allocation is required for primitives consisting of multiple kernels in dynamic case
    {
        if (_impl == nullptr)
            return;
        const auto& ibuf_layouts = _impl->get_internal_buffer_layouts();
        if (ibuf_layouts.empty())
            return;

        for (size_t i = 0; i < ibuf_layouts.size(); ++i) {
            if (i < _intermediates_memory.size() && ibuf_layouts[i].bytes_count() <= max_intermediates_memory_sizes[i]) {
                // can reuse
                _intermediates_memory[i] = _network.get_engine().reinterpret_buffer(*_intermediates_memory[i], ibuf_layouts[i]);
            } else {
                if (i < _intermediates_memory.size()) {
                    _intermediates_memory[i] = allocate_internal_buffer(i);
                    max_intermediates_memory_sizes[i] = _intermediates_memory[i]->size();
                } else {
                    // i-th layout has not been allocated yet
                    _intermediates_memory.push_back(allocate_internal_buffer(i));
                    max_intermediates_memory_sizes.push_back(_intermediates_memory[i]->size());
                }
            }
        }
    }
}

bool primitive_inst::update_impl() {
    GPU_DEBUG_PROFILED_STAGE(instrumentation::pipeline_stage::update_implementation);
    auto prev_impl_str =  _impl != nullptr ? _impl->get_kernel_name() : "nullptr";

    auto update_shape_info = [this, prev_impl_str](const kernel_impl_params& params) {
        mem_lock<int32_t> lock(_shape_info_memory, _network.get_stream());
        size_t offset = 0;
        for (size_t i = 0; i < _node->get_dependencies().size(); i++) {
            if (_node->get_dependency(i).get_output_layout().is_dynamic()) {
                auto input_shape = _node->type()->extend_input_shape_to_6d(params, i);
                for (size_t j = 0; j < input_shape.size(); j++)
                    lock[offset++] = static_cast<int32_t>(input_shape[j]);
            }
        }

        for (size_t i = 0; i < _node->get_output_layouts().size(); i++) {
            if (_node->get_output_layout(i).is_dynamic()) {
                auto output_shape = _node->type()->extend_output_shape_to_6d(params, i);
                for (size_t j = 0; j < output_shape.size(); j++)
                    lock[offset++] = static_cast<int32_t>(output_shape[j]);
            }
        }
        std::stringstream s;
        s << "shapes: ";
        for (size_t i = 0; i < offset; i++)
            s << lock[i] << " ";
        GPU_DEBUG_TRACE_DETAIL << id() << ": update dynamic impl " << prev_impl_str << " to new shape: " << s.str() << std::endl;
    };

    if (!_node->is_type<data>() && !(_node->is_type<mutable_data>() && _node->get_dependencies().empty())) {
        // Update param if fake_alignment is available
        auto updated_params = _node->type()->get_fake_aligned_params(*_impl_params);
        auto& cache = get_network().get_program()->get_implementations_cache();
        std::shared_ptr<primitive_impl> cached_impl = nullptr;
        {
            cached_impl = cache.get(updated_params);
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
                auto& compilation_context = get_network().get_program()->get_compilation_context();
                compilation_context.push_task(updated_params.hash(), [this, &compilation_context, updated_params]() {
                    if (compilation_context.is_stopped())
                        return;
                    auto _program = get_network().get_program();
                    auto& cache = _program->get_implementations_cache();
                    {
                        // Check existense in the cache one more time as several iterations of model execution could happens and multiple compilation
                        // tasks created for same shapes
                        if (cache.has(updated_params))
                            return;
                    }

                    auto impl = _node->type()->choose_impl(*_node, updated_params);
                    auto kernels = _program->get_kernels_cache().compile(impl->get_kernels_source());
                    impl->set_kernels(kernels);
                    cache.add(updated_params, impl->clone());
                });
                _impl = _dynamic_impl->clone();
                _impl->update_dispatch_data(*_impl_params);

                update_shape_info(*_impl_params);
            } else {
                _impl = _node->type()->choose_impl(*_node, updated_params);
                auto& kernels_cache = get_network().get_program()->get_kernels_cache();
                auto kernels = kernels_cache.compile(_impl->get_kernels_source());
                _impl->set_kernels(kernels);
                cache.add(updated_params, _impl->clone());

                auto new_impl_str = _impl != nullptr ? _impl->get_kernel_name() : "nullptr";
                GPU_DEBUG_TRACE_DETAIL << id() << ": update impl from " << prev_impl_str << " to " << new_impl_str << std::endl;
            }
        }

        reset_shape_change();
    }
    // impl is replaced
    return true;
}

event::ptr primitive_inst::execute(const std::vector<event::ptr>& events) {
    const auto primitive_id = id();
    OPENVINO_ASSERT(_has_valid_input, primitive_id, " has invalid/unset input");
    GPU_DEBUG_GET_INSTANCE(debug_config);

    std::vector<event::ptr> dependencies;
    if (is_dynamic()) {
        OPENVINO_ASSERT(_node != nullptr, "[GPU] Invalid primitive_inst object for dynamic shapes case: program_node can't be null");
        update_shape();
        if (_impl_params->output_layouts[0].bytes_count() == 0) {
            auto ev = get_network().get_stream().create_user_event(true);
            return ev;
        }

        if (!is_valid_fusion()) {
            auto subgraph = get_unfused_subgraph();

            for (auto& d : _deps) {
                if (!d.first->get_node().is_type<data>()) {
                    auto allocated_mem = d.first->output_memory_ptr();
                    auto actual_input_layout = d.first->get_output_layout();
                    auto& engine = _network.get_engine();
                    // Need to use actual layout, not the fake aligned memory layout
                    auto actual_mem = engine.reinterpret_buffer(*allocated_mem, actual_input_layout);
                    subgraph->set_input_data(d.first->id(), actual_mem);
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
            if (update_impl()) {
                auto ev = update_weights();
                if (ev)
                    dependencies.push_back(ev);
                realloc_if_needed();
            }
        }
    }

    OPENVINO_ASSERT(_impl_params->get_output_layout().is_static(),
                    "[GPU] Can't execute ", primitive_id, " primitive as output layout is dynamic in runtime");

    OPENVINO_ASSERT(_impl != nullptr, "[GPU] Implementation is nullptr for ", primitive_id,  " primitive");

    // Output buffer may be changed under the following conditions, so we need to set args to kernel on each iteration
    if (is_dynamic() || has_mutable_input() || is_output()) {
        set_arguments();
    }
    on_execute();

    GPU_DEBUG_TRACE << id() << ": execute " << _impl->get_kernel_name() << std::endl;

    if (_exec_deps.empty() && dependencies.empty()) {
        dependencies = events;
    } else {
        auto queue_type = get_network().get_stream().get_queue_type();
        if (queue_type == QueueTypes::out_of_order) {
            dependencies.reserve(dependencies.size() + _exec_deps.size());
            for (auto& input : _exec_deps) {
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

    {
        GPU_DEBUG_PROFILED_STAGE(instrumentation::pipeline_stage::inference);
        auto ev = _impl->execute(dependencies, *this);

        GPU_DEBUG_IF(!debug_config->dump_profiling_data.empty()) {
            get_network().get_stream().wait_for_events({ev});
        }

        return ev;
    }
}

void primitive_inst::set_arguments() {
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
    , _output_changed(false)
    , _mem_allocated(false) {}

primitive_inst::primitive_inst(network& network, program_node const& node, bool allocate_memory)
    : _network(network)
    , _node(&node)
    , _node_output_layout(node.get_output_layout())
    , _impl_params(node.get_kernel_impl_params())
    , _impl(node.get_selected_impl() ? node.get_selected_impl()->clone() : nullptr)
    , _dynamic_impl(nullptr)
    , _outputs({memory::ptr()})
    , _output_changed(false)
    , _mem_allocated(allocate_memory)
    , _is_dynamic(node.is_dynamic() || node.generates_dynamic_output())
    , _type(node.type())
    , _id(node.id())
    , _org_id(node.get_org_primitive_id())
    , _is_input(node.is_input())
    , _is_output(node.is_output())
    , _inputs_memory_count(node.get_primitive()->input_size())
    , _outputs_memory_count(node.get_primitive()->output_size())
    , _fused_mem_count(node.get_fused_inputs_count())
    , _fused_mem_offset(_fused_mem_count > 0 ? node.get_fused_primitives()[0].dep_start_idx : 0)
    , _can_be_optimized(node.can_be_optimized())
    , _can_share_buffer(node.can_share_buffer())
    , _is_constant(node.is_constant()) {
    if (allocate_memory) {
        // In case when output is mutable_data primitive, and other users dependencies are only used for
        // suychronization, The output memory of such primitive will be fused with mutable_data
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
        if (_impl->is_dynamic()) {
            _dynamic_impl = _impl->clone();
            // Actual shape info layout is the following:
            // input_0 -> input_1, ..., fused_dep_0, fused_dep1, ..., output_0, output_1, ...
            // For each tensor we save 6 dimensions if [bfwzyx] order
            const int64_t buffers_count = _node->get_dependencies().size() + _node->get_outputs_count();
            const size_t tensor_dims_count = 6;
            const int64_t shape_elements = buffers_count * tensor_dims_count;
            _shape_info_memory = _network.get_engine().allocate_memory(layout{{shape_elements}, data_types::i32, format::bfyx});
        }
    }

    if (_outputs[0])
        max_output_layout_size = _outputs[0]->get_layout().get_tensor().count();
}

memory::ptr primitive_inst::allocate_internal_buffer(size_t idx) {
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
    if (input_device_mem && (available_device_mem_size - (int64_t)layout.bytes_count() >= 0)) {
        alloc_type = engine.get_preferred_memory_allocation_type();
    } else {
        alloc_type = engine.get_lockable_preferred_memory_allocation_type();
    }
    return engine.allocate_memory(layout, alloc_type);
}

void primitive_inst::allocate_internal_buffers(void) {
    if (_impl == nullptr || _outputs.empty() || _outputs[0] == nullptr)
        return;
    const auto& ibuf_layouts = _impl->get_internal_buffer_layouts();
    if (ibuf_layouts.empty())
        return;

    // allocate intermediate memory for the updated layout of buffer
    std::vector<memory::cptr> intermediates_memory;
    for (size_t i = 0; i < ibuf_layouts.size(); ++i) {
        intermediates_memory.push_back(allocate_internal_buffer(i));
        max_intermediates_memory_sizes.push_back(intermediates_memory[i]->size());
    }
    _intermediates_memory = intermediates_memory;
}

event::ptr primitive_inst::update_weights() {
    GPU_DEBUG_PROFILED_STAGE(instrumentation::pipeline_stage::update_weights);
    if (!_impl)
        return nullptr;

    bool weightable_node = _node->is_type<fully_connected>() || _node->is_type<convolution>() || _node->is_type<deconvolution>();
    if (!weightable_node)
        return nullptr;

    auto& weights_params = _impl->_weights_reorder_params;
    bool requires_reorder = weights_params.engine != kernel_selector::GenericKernelParams::Engine::NONE &&
                            (!_impl_params->reordered_weights || _impl_params->reordered_weights->get_layout() != from_weights_tensor(weights_params.dest));
    if (requires_reorder) {
        GPU_DEBUG_PROFILED_STAGE_CACHE_HIT(false);
        auto weights_idx = _node->get_primitive()->input.size();
        auto original_weights_memory = dep_memory_ptr(weights_idx);
        auto original_layout = original_weights_memory->get_layout();
        layout expected_layout = from_weights_tensor(weights_params.dest);
        auto& engine = _network.get_engine();

        auto get_kernel_key = [&]() -> size_t {
            auto seed = _node->get_primitive()->hash();
            seed = hash_combine(seed, expected_layout.hash());
            seed = hash_combine(seed, original_layout.hash());
            return seed;
        };

        cldnn::kernel::ptr kernel = nullptr;
        auto kernel_key = get_kernel_key();
        auto& cache = get_network().get_in_mem_kernels_cache();
        if (cache.has(kernel_key)) {
            GPU_DEBUG_TRACE_DETAIL << id() << ": reorder weights (cached) from " << original_layout.to_short_string()
                                    << " to " << expected_layout.to_short_string() << std::endl;
            GPU_DEBUG_PROFILED_STAGE_CACHE_HIT(true);
            kernel = cache.get(kernel_key);
        } else {
            GPU_DEBUG_TRACE_DETAIL << id() << ": reorder weights from " << original_layout.to_short_string()
                                    << " to " << expected_layout.to_short_string() << std::endl;
            auto& kernels_cache = get_network().get_program()->get_kernels_cache();
            auto kernels = kernels_cache.compile({weights_params.clKernel->code.kernelString});
            OPENVINO_ASSERT(kernels.size() == 1, "The output of kernel compile has issue");
            kernel = kernels.begin()->second;
            cache.add(kernel_key, kernel);
        }

        auto& stream = get_network().get_stream();

        bool can_reuse = _impl_params->reordered_weights != nullptr && _impl_params->reordered_weights->size() <= expected_layout.bytes_count();
        if (can_reuse) {
            GPU_DEBUG_TRACE_DETAIL << id() << ": reuse weights memory" << std::endl;
            _impl_params->reordered_weights = engine.reinterpret_buffer(*_impl_params->reordered_weights, expected_layout);
        } else {
            auto alloc_type = engine.get_preferred_memory_allocation_type();
            _impl_params->reordered_weights = engine.allocate_memory(expected_layout, alloc_type);
        }

        kernel_arguments_data args;
        args.inputs.push_back(original_weights_memory);
        args.outputs.push_back(_impl_params->reordered_weights);
        stream.set_arguments(*kernel, weights_params.clKernel->params, args);
        auto ev = stream.enqueue_kernel(*kernel, weights_params.clKernel->params, args, {}, true);

        GPU_DEBUG_GET_INSTANCE(debug_config);
        GPU_DEBUG_IF(!debug_config->dump_profiling_data.empty()) {
            stream.wait_for_events({ev});
        }

        return ev;
    } else {
        // If kernel doesn't says that it doesn't require weights reorder, but weights were reordered previously, then
        // incorrect memory buffer may be assigned, so reset cached weights for such case
        if (weights_params.engine == kernel_selector::GenericKernelParams::Engine::NONE) {
            _impl_params->reordered_weights.reset();
        }
    }
    GPU_DEBUG_PROFILED_STAGE_CACHE_HIT(true);

    return nullptr;
}

static bool user_requesting_mem_reuse_false(const program_node& node) {
    for (auto& user : node.get_users()) {
        if (user->is_dynamic())
            return true;
        if ((user->get_selected_impl() != nullptr) && (user->get_selected_impl()->can_reuse_memory == false)) {
            return true;
        } else if (user->get_selected_impl() == nullptr) {
            if (user_requesting_mem_reuse_false(*user)) {
                return true;
            }
        }
    }
    return false;
}

memory::ptr primitive_inst::allocate_output(engine& _engine, memory_pool& pool, const program_node& _node, const kernel_impl_params& impl_params,
                                            uint32_t net_id, bool is_internal, size_t idx) {
    auto get_memory_from_pool = [&](engine& _engine, const layout& layout, const primitive_id id, std::set<primitive_id> dependencies,
            allocation_type type, bool reusable) {
        OPENVINO_ASSERT(!layout.is_dynamic() || layout.has_upper_bound(), "[GPU] Can't allocate output for dynamic layout without upper bound");
        // Use layout with max tensor for dynamic shape with upper bound
        auto static_layout = cldnn::layout(layout.data_type, layout.format, layout.get_tensor(), layout.data_padding);
        if (_node.get_program().get_config().get_property(ov::intel_gpu::enable_memory_pool))
            return pool.get_memory(static_layout, id, net_id, dependencies, type, reusable);
        return pool.get_memory(static_layout, type);
    };


    auto layout = impl_params.get_output_layout(idx);
    OPENVINO_ASSERT(layout.is_static() || layout.has_upper_bound(), "[GPU] Can't allocate output for dynamic layout");
    auto device_mem_acc = [&](size_t a, const cldnn::layout& l) {
        // Input shape may be dynamic is some cases (shape_of). It means that output shape of node doesn't depend on input shape
        // and out memory can be allocated on program build stage.
        if (l.is_static())
            return a + l.bytes_count();

        return a;
    };

    bool usm_device_allocatable = true;
    const auto& total_device_input_mem_size = std::accumulate(impl_params.input_layouts.begin(), impl_params.input_layouts.end(), (uint64_t)0, device_mem_acc);
    if (total_device_input_mem_size > _engine.get_device_info().max_global_mem_size)
        usm_device_allocatable = false;

    bool memory_reuse_by_user = true;

    if (user_requesting_mem_reuse_false(_node)) {
        memory_reuse_by_user = false;
    }

    // For outputs, cpu prim we want to have lockable alloc type
    // Also if the successor of a node is an cpu, then memory needs to be lockable.
    bool is_cpu = _node.get_selected_impl() ? _node.get_selected_impl()->is_cpu() : false;
    auto use_lockable_memory = is_output_buffer(_node) || is_cpu || is_any_user_cpu(_node.get_users()) ||
                               !_engine.supports_allocation(allocation_type::usm_device);
    const auto& lockable_mem_type = _engine.get_lockable_preferred_memory_allocation_type(layout.format.is_image_2d());

    // If this node is to be used as shape infer, it needs to copy data to be used by shape infer.
    auto alloc_type = use_lockable_memory ? lockable_mem_type
                    : !usm_device_allocatable ? lockable_mem_type :
                      !_node.is_shape_infer_dep() ? allocation_type::usm_device : lockable_mem_type;

    if ((is_internal && (_node.can_be_optimized() || _node.is_type<generic_layer>())) || (memory_reuse_by_user == false)) {
        GPU_DEBUG_LOG << "[" << _node.id() << ": output]" << std::endl;
        // Use usm_device memory for weights reordering
        if (is_internal && _node.is_type<generic_layer>() && _engine.supports_allocation(allocation_type::usm_device))
            alloc_type = allocation_type::usm_device;
        return get_memory_from_pool(_engine,
                layout,
                _node.id(),
                _node.get_memory_dependencies(),
                alloc_type,
                false);
    } else if (is_internal && _node.is_output() && _node.is_type<generic_layer>() &&
            _engine.supports_allocation(allocation_type::usm_device) && usm_device_allocatable) {
        GPU_DEBUG_LOG << "[" << _node.id() << ": output]" << std::endl;
        return _engine.allocate_memory(layout, allocation_type::usm_device, false);
    } else if (is_internal && !_node.is_output() && _node.is_type<input_layout>()) {
        // Skip memory reset for input_layout primitives, since data will be copied from cldnn::data primitive
        // or just reuse primitive's memory
        GPU_DEBUG_LOG << "[" << _node.id() << ": constant]" << std::endl;
        return _engine.allocate_memory(layout, alloc_type, false);
    } else if (is_internal || (!_node.can_share_buffer()) || _node.can_be_optimized() || _node.is_output()) {
        GPU_DEBUG_LOG << "[" << _node.id() << ": output]" << std::endl;
        return _engine.allocate_memory(layout, alloc_type);
    } else {
        return get_memory_from_pool(_engine,
                layout,
                _node.id(),
                _node.get_memory_dependencies(),
                alloc_type,
                true);
    }
}

std::vector<memory::ptr> primitive_inst::allocate_outputs(kernel_impl_params* updated_params) {
    std::vector<memory::ptr> outputs;
    for (size_t i = 0; i < get_node().get_outputs_count() ; ++i) {
        outputs.push_back(allocate_output(get_network().get_engine(), _network.get_memory_pool(),
                         *_node, (updated_params != nullptr) ? *updated_params : *_impl_params,
                         get_network_id(), _network.is_internal(), i));
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

        std::vector<primitive_id> dep_ids;
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
            dep_ids.push_back(dep.first->id());
        }

        // Create the primitive itself
        t.add_primitive(std::const_pointer_cast<primitive>(_node->get_primitive()));
        dep_ids.push_back(_node->id());

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
                if (std::find_if(dep_ids.begin(), dep_ids.end(),
                                 [&](const primitive_id& pid) {
                                     return pid == in.pid;
                                 }) == dep_ids.end()) {
                    size_t dep_id = fd.dep_start_idx;
                    in = _node->get_dependency(dep_id).id();
                }
            }
            t.add_primitive(prim);
            dep_ids.push_back(prim->id);
        }
        // Samely, need to update dependency of the current fused nodes' input primitive ids with those in the current program
        auto prim_of_fused_node = std::const_pointer_cast<primitive>(_impl_params->desc);
        for (size_t i = 0; i < prim_of_fused_node->input.size(); ++i) {
            auto& in = prim_of_fused_node->input[i];
            if (std::find_if(dep_ids.begin(), dep_ids.end(),
                             [&](const primitive_id& pid) {
                                 return pid == in.pid;
                             }) == dep_ids.end()) {
                in = _node->get_dependency(i).id();
            }
        }
        ExecutionConfig subgraph_config{
            ov::intel_gpu::allow_static_input_reorder(true),
            ov::intel_gpu::allow_new_shape_infer(true)
        };
        auto prog = program::build_program(get_network().get_engine(), t, subgraph_config, true, false);

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
        if (fd.is_type<eltwise>()) {
            fused_eltwise_prims.push_back(fd);
        }
    }

    if (fused_eltwise_prims.empty())
        return true;

    auto out_pshape = _impl_params->get_output_layout().get_partial_shape();
    for (auto& fd : fused_eltwise_prims) {
        auto dep_idx = fd.dep_start_idx;
        OPENVINO_ASSERT(fd.total_num_deps == 2, "[GPU] Unexpected count of dependencies in dynamic fusion for eltwise");
        OPENVINO_ASSERT(_deps.size() > dep_idx, "[GPU] Invalid fused dependency idx");
        auto dep = _deps[dep_idx];

        auto dep_pshape = dep.first->_impl_params->get_output_layout().get_partial_shape();
        auto merged_shape = out_pshape;
        auto can_broadcast = ov::PartialShape::broadcast_merge_into(merged_shape, dep_pshape, fd.typed_desc<eltwise>()->broadcast_spec);

        // We check that broadcasting of extra input is possible and it doesn't change output shape. If it output shape is changed, then
        // some dimension of dep_pshape is greater than out_pshape
        if (!can_broadcast || merged_shape != out_pshape)
            return false;
    }

    return true;
}

void primitive_inst::add_profiling_data(instrumentation::pipeline_stage stage, bool cache_hit, int64_t time) {
    instrumentation::perf_counter_key key {
            _network.get_input_layouts(),
            _impl_params->input_layouts,
            _impl_params->output_layouts,
            get_implementation_name(),
            stage,
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
    ob.setKernlImplParams(_impl_params.get());

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
    auto users = get_node().get_users();
    bool is_output_event = is_any_user_cpu(users) || get_node().is_output();
    ob << is_output_event;

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

    IE_THROW() << "[get_index_in_deps]: not found in _deps";
}

void primitive_inst::load(cldnn::BinaryInputBuffer& ib) {
    _impl_params->load(ib);
    ib.setKernlImplParams(_impl_params.get());

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
    ib >> _is_output_event;

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

            allocation_type _allocation_type;
            ib >> make_data(&_allocation_type, sizeof(_allocation_type));
            allocation_types.emplace_back(_allocation_type);
        }
    }

    bool can_reuse_memory;
    ib >> can_reuse_memory;

    std::set<primitive_id> _node_mem_deps;
    ib >> _node_mem_deps;

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

        _intermediates_memory[i] = get_network().get_engine().allocate_memory(ibuf_layout, _allocation_type);
    }

    bool has_impl;
    ib >> has_impl;
    if (has_impl) {
        _impl.release();
        ib >> _impl;
    }
}

}  // namespace cldnn
