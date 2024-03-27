// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/runtime/make_tensor.hpp"
#include "openvino/core/preprocess/input_tensor_info.hpp"
#include "openvino/core/parallel.hpp"
#include "openvino/core/validation_util.hpp"

#include "intel_gpu/primitives/kv_cache.hpp"
#include "intel_gpu/plugin/usm_host_tensor.hpp"
#include "intel_gpu/plugin/sync_infer_request.hpp"
#include "intel_gpu/plugin/remote_context.hpp"
#include "intel_gpu/plugin/remote_tensor.hpp"
#include "intel_gpu/plugin/compiled_model.hpp"
#include "intel_gpu/plugin/variable_state.hpp"
#include "intel_gpu/plugin/multi_tensor_variable_state.hpp"
#include "intel_gpu/runtime/memory.hpp"
#include "intel_gpu/runtime/memory_caps.hpp"
#include "intel_gpu/runtime/internal_properties.hpp"
#include "intel_gpu/runtime/itt.hpp"
#include "intel_gpu/runtime/debug_configuration.hpp"

#include <algorithm>
#include <iterator>
#include <memory>
#include <string>
#include <map>
#include <functional>
#include <utility>

namespace {

inline bool can_use_usm_host(const cldnn::engine& engine) {
    auto can_use_usm = engine.use_unified_shared_memory();

    if (engine.get_device_info().gfx_ver.major == 12 && engine.get_device_info().gfx_ver.minor == 60) {
        // WA: Disable USM host memory for infer request`s tensors for PVC as
        // it has performance issues in case of host <-> device data transfers inside kernels
        GPU_DEBUG_TRACE << "Do not use usm_host for performance issue" << std::endl;
        can_use_usm = false;
    }

    return can_use_usm;
}

bool is_convert_required(ov::element::Type src_et, ov::element::Type dst_et) {
    return src_et != dst_et && !(dst_et == ov::element::boolean && src_et == ov::element::u8);
}

bool same_host_mem(cldnn::memory::cptr memory, const uint8_t* host_ptr) {
    const uint8_t* device_ptr = memory->get_allocation_type() == cldnn::allocation_type::usm_host ?
                                static_cast<uint8_t*>(memory->get_internal_params().mem) : nullptr;
    return device_ptr == host_ptr;
}

inline bool all_remote_buffers(const std::vector<ov::SoPtr<ov::ITensor>>& tensors) {
    return std::all_of(tensors.begin(), tensors.end(), [](const ov::SoPtr<ov::ITensor>& tensor) {
        if (auto remote_ptr = std::dynamic_pointer_cast<ov::intel_gpu::RemoteTensorImpl>(tensor._ptr)) {
            return !remote_ptr->is_surface();
        }
        return false;
    });
}

inline bool all_remote_surfaces(const std::vector<ov::SoPtr<ov::ITensor>>& tensors) {
    return std::all_of(tensors.begin(), tensors.end(), [](const ov::SoPtr<ov::ITensor>& tensor) {
        if (auto remote_ptr = std::dynamic_pointer_cast<ov::intel_gpu::RemoteTensorImpl>(tensor._ptr)) {
            return remote_ptr->is_surface();
        }
        return false;
    });
}

inline bool all_host_tensors(const std::vector<ov::SoPtr<ov::ITensor>>& tensors) {
    return std::all_of(tensors.begin(), tensors.end(), [](const ov::SoPtr<ov::ITensor>& tensor) {
        return std::dynamic_pointer_cast<ov::intel_gpu::RemoteTensorImpl>(tensor._ptr) == nullptr;
    });
}

}  // namespace

namespace ov {
namespace intel_gpu {

// ----------------------------------------------------------------------------------------------- //
// ---------------------------- OpenVINO API impl ------------------------------------------------ //
// ----------------------------------------------------------------------------------------------- //

SyncInferRequest::SyncInferRequest(const std::shared_ptr<const CompiledModel>& compiled_model)
    : ov::ISyncInferRequest(compiled_model)
    , m_graph(compiled_model->get_graph(0))
    , m_context(std::static_pointer_cast<RemoteContextImpl>(compiled_model->get_context_impl()))
    , m_shape_predictor(new cldnn::ShapePredictor(&m_graph->get_engine(), m_graph->get_config().get_property(ov::intel_gpu::buffers_preallocation_ratio)))
    , m_enable_profiling(m_graph->get_config().get_property(ov::enable_profiling))
    , m_use_external_queue(m_graph->use_external_queue()) {
    GPU_DEBUG_GET_INSTANCE(debug_config);
    GPU_DEBUG_IF(debug_config->mem_preallocation_params.is_initialized) {
        auto& mem_preallocation_params = debug_config->mem_preallocation_params;
        m_shape_predictor.reset(
            new cldnn::ShapePredictor(&m_graph->get_engine(),
                                      mem_preallocation_params.next_iters_preallocation_count,
                                      mem_preallocation_params.max_per_iter_size,
                                      mem_preallocation_params.max_per_dim_diff,
                                      mem_preallocation_params.buffers_preallocation_ratio));
    }

    init_mappings();
    allocate_inputs();
    allocate_outputs();
    allocate_states();
}

void SyncInferRequest::infer() {
    OV_ITT_SCOPED_TASK(itt::domains::intel_gpu_plugin, "SyncInferRequest::infer");
    setup_stream_graph();
    std::lock_guard<std::mutex> lk(m_graph->get_mutex());
    enqueue();
    wait();
}

std::vector<ov::ProfilingInfo> SyncInferRequest::get_profiling_info() const {
    OPENVINO_ASSERT(m_enable_profiling, "[GPU] Profiling data was not collected: please check that ov::enable_profiling property was set to true");
    return m_graph->get_profiling_info();
}

std::vector<ov::SoPtr<ov::IVariableState>> SyncInferRequest::query_state() const {
    std::vector<ov::SoPtr<ov::IVariableState>> ret{};
    for (const auto& pair : m_variables) {
        ret.emplace_back(pair.second, nullptr);
    }
    return ret;
}

void SyncInferRequest::set_tensor(const ov::Output<const ov::Node>& port, const ov::SoPtr<ov::ITensor>& tensor) {
    OV_ITT_SCOPED_TASK(itt::domains::intel_gpu_plugin, "SyncInferRequest::set_tensor");
    const auto& port_info = find_port(port);
    size_t port_index = port_info.idx;
    const auto& shape = port.get_partial_shape();

    OPENVINO_ASSERT(tensor != nullptr, "[GPU] Failed to set empty tensor to port with index: \'", port_index, "\'");
    OPENVINO_ASSERT(port.get_element_type() == tensor->get_element_type(),
                    "[GPU] Mismatch tensor and port type: ", port.get_element_type(), " vs ", tensor->get_element_type());
    OPENVINO_ASSERT(shape.compatible(ov::PartialShape(tensor->get_shape())) || tensor->get_shape() == ov::Shape {0} || port.get_partial_shape().is_dynamic(),
                    "[GPU] The tensor size is not equal to model, can't set input tensor with index: ",
                    port_index,
                    ", because model input (shape=",
                    shape,
                    ") and tensor (shape=",
                    tensor->get_shape(),
                    ") are incompatible");

    auto update_tensors_maps = [](size_t port_index,
                                  std::unordered_map<size_t, ov::intel_gpu::TensorWrapper>& user_tensors,
                                  std::unordered_map<size_t, ov::intel_gpu::TensorWrapper>& plugin_tensors,
                                  const ov::SoPtr<ov::ITensor>& tensor) {
        auto current_tensor_owner = user_tensors[port_index].owner;
        auto is_same_tensor = user_tensors[port_index].ptr == tensor._ptr;

        // Keep PLUGIN as a tensor owner if current user's tensor owner is PLUGIN and underlying tensor pointer is not changed
        auto new_tensor_owner = current_tensor_owner == TensorOwner::PLUGIN && is_same_tensor ? TensorOwner::PLUGIN
                                                                                              : TensorOwner::USER;

        user_tensors[port_index] = { tensor._ptr, new_tensor_owner };

        // We need to properly handle PLUGIN -> USER ownership change to prevent invalid PLUGIN's ush_host buffer sharing,
        // so remove plugin's tensor to reallocate it in prepare_input() method
        if (current_tensor_owner == TensorOwner::PLUGIN && new_tensor_owner == TensorOwner::USER) {
            if (plugin_tensors.count(port_index) && std::dynamic_pointer_cast<RemoteTensorImpl>(plugin_tensors[port_index].ptr)->is_shared())
                plugin_tensors.erase(plugin_tensors.find(port_index));
        }
    };

    bool is_input = port_info.type == ov::ISyncInferRequest::FoundPort::Type::INPUT;
    if (is_input) {
        update_tensors_maps(port_index, m_user_inputs, m_plugin_inputs, tensor);
    } else {
        update_tensors_maps(port_index, m_user_outputs, m_plugin_outputs, tensor);
    }

    ov::ISyncInferRequest::set_tensor(port, tensor);
}

void SyncInferRequest::set_tensors_impl(const ov::Output<const ov::Node> port, const std::vector<ov::SoPtr<ov::ITensor>>& tensors) {
    if (tensors.size() == 1) {
        return set_tensor(port, tensors[0]);
    }
    const auto& port_info = find_port(port);
    bool is_input = port_info.type == ov::ISyncInferRequest::FoundPort::Type::INPUT;
    OPENVINO_ASSERT(is_input, "[GPU] set_tensors_impl is not supported for output port");

    bool is_remote = all_remote_buffers(tensors) || all_remote_surfaces(tensors);
    bool is_host = all_host_tensors(tensors);

    OPENVINO_ASSERT(is_host || is_remote, "[GPU] Incorrect input blobs. All blobs must be of the same type");

    size_t port_index = port_info.idx;
    OPENVINO_ASSERT(m_input_ports_map.count(port_index) != 0, "[GPU] Cannot find input tensors for port ", port, " with index ", port_index);
    const auto& tensor = m_input_ports_map.at(port_index).get_tensor_ptr();
    m_batched_tensors[tensor] = tensors;
}

ov::SoPtr<ov::ITensor> SyncInferRequest::get_tensor(const ov::Output<const ov::Node>& port) const {
    const auto& port_info = find_port(port);
    bool is_input = port_info.type == ov::ISyncInferRequest::FoundPort::Type::INPUT;
    size_t port_index = port_info.idx;
    if (is_input) {
        OPENVINO_ASSERT(m_user_inputs.count(port_index) == 1, "[GPU] Input tensor with index ", port_index, " is not found");
        return { m_user_inputs.at(port_index).ptr, nullptr };
    } else {
        OPENVINO_ASSERT(m_user_outputs.count(port_index) == 1, "[GPU] Output tensor with index ", port_index, " is not found");
        return { m_user_outputs.at(port_index).ptr, nullptr };
    }
}

void SyncInferRequest::check_tensors() const {
    const auto& inputs = get_compiled_model()->inputs();
    for (size_t i = 0; i < inputs.size(); i++) {
        if (!is_batched_input(inputs[i]))
            check_tensor(inputs[i], get_tensor_ptr(inputs[i]));
    }
    const auto& outputs = get_compiled_model()->outputs();
    for (size_t i = 0; i < outputs.size(); i++) {
        check_tensor(outputs[i], get_tensor_ptr(outputs[i]));
    }
}

// ----------------------------------------------------------------------------------------- //
// ---------------------------- internal pipeline stages ----------------------------------- //
// ----------------------------------------------------------------------------------------- //
void SyncInferRequest::set_task_executor(const std::shared_ptr<ov::threading::ITaskExecutor>& task_executor) {
    m_stream_executor = std::dynamic_pointer_cast<ov::threading::IStreamsExecutor>(task_executor);
}

void SyncInferRequest::enqueue_notify() {
    m_graph->wait(Graph::Stage::EXECUTE);
    enqueue();
}

void SyncInferRequest::wait_notify() {
    wait();
    m_graph->notify(Graph::Stage::EXECUTE);
}

void SyncInferRequest::enqueue() {
    // set input and output memory from request blob maps
    // into the network object primitives
    std::vector<cldnn::event::ptr> dependencies;

    for (const auto& it : m_input_ports_map) {
        size_t port_idx = it.first;
        const auto& port = it.second;

        if (m_batched_tensors.count(port.get_tensor_ptr()) > 0) {
            auto events = prepare_batched_input(port_idx, port, m_batched_tensors.at(port.get_tensor_ptr()));
            std::move(events.begin(), events.end(), std::back_inserter(dependencies));
        } else {
            cldnn::primitive_id internal_name = m_graph->input_port_index_to_internal(port_idx)[0];
            auto events = prepare_input(internal_name, port_idx, port, m_user_inputs.at(port_idx));
            std::move(events.begin(), events.end(), std::back_inserter(dependencies));
        }
    }

    for (const auto& it : m_output_ports_map) {
        size_t port_idx = it.first;
        const auto& port = it.second;

        auto events = prepare_output(port_idx, port, m_user_outputs.at(port_idx));
        std::move(events.begin(), events.end(), std::back_inserter(dependencies));
    }

    for (const auto& it : m_variables) {
        const auto& name = it.first;
        const auto& variable = it.second;
        prepare_state(name, variable);
    }

    auto network = m_graph->get_network();
    network->set_shape_predictor(m_shape_predictor);

    m_internal_outputs.clear();
    m_internal_outputs = network->execute(dependencies);

    // If dump layers path is set, only runs first inference.
    GPU_DEBUG_GET_INSTANCE(debug_config);
    GPU_DEBUG_IF(debug_config->dump_layers_path.length() > 0 && debug_config->dump_iteration.empty()) {
        GPU_DEBUG_INFO << "Only run first inference to dump layers." << std::endl;
        exit(0);
    }
}

void SyncInferRequest::wait() {
    OV_ITT_SCOPED_TASK(itt::domains::intel_gpu_plugin, "SyncInferRequest::wait");
    OPENVINO_ASSERT(!m_internal_outputs.empty(), "[GPU] Inference was not started!\n");

    auto& network = *m_graph->get_network();

    // wait for completion & collect outputs as requested by the model
    // for in_order_queue, it is enough to call finish only once
    bool do_sync_per_output = (network.get_stream().get_queue_type() == QueueTypes::in_order) ? false : true;
    if (!do_sync_per_output)
        network.get_stream().finish();

    std::vector<cldnn::event::ptr> copy_events;

    for (const auto& it : m_output_ports_map) {
        size_t port_idx = it.first;
        const auto& port = it.second;
        cldnn::primitive_id internal_name = m_output_names_map.at(port_idx);
        auto output_memory = m_internal_outputs.at(internal_name).get_memory(do_sync_per_output);
        auto output_layout = m_internal_outputs.at(internal_name).get_layout();

        if (output_memory) {
            OV_ITT_SCOPED_TASK(itt::domains::intel_gpu_plugin, "SyncInferRequest::wait::reinterpret_memory");
            OPENVINO_ASSERT(!output_memory->get_layout().data_padding, "[GPU] Unexpected padding in output buffer");
            output_memory = m_graph->get_engine().reinterpret_buffer(*output_memory, output_layout);
            GPU_DEBUG_TRACE_DETAIL << internal_name << " model output with index " << port_idx << ": " << output_memory->buffer_ptr() << std::endl;
        }

        OPENVINO_ASSERT(m_user_outputs.count(port_idx) > 0, "[GPU] Output index ", port_idx, " is not found in output tensors map");
        auto output_tensor_wrapper = m_user_outputs.at(port_idx);
        auto output_tensor = output_tensor_wrapper.ptr;
        auto remote_ptr = std::dynamic_pointer_cast<RemoteTensorImpl>(output_tensor);
        bool is_remote = remote_ptr != nullptr;
        bool is_dynamic = port.get_partial_shape().is_dynamic();

        if (is_remote) {
            GPU_DEBUG_TRACE_DETAIL << internal_name << " handle output tensor (remote) with index: " << port_idx << ": "
                                   << remote_ptr->get_original_memory()->buffer_ptr() << std::endl;
        } else {
            GPU_DEBUG_TRACE_DETAIL << internal_name << " handle output tensor (host) with index: " << port_idx << ": "
                                   << output_tensor->data() << std::endl;
        }

        OPENVINO_ASSERT(output_tensor_wrapper.owner == TensorOwner::PLUGIN || is_dynamic || output_tensor_wrapper.actual_size >= output_memory->size(),
                        "[GPU] Output port is static and output tensor set by user has smaller size (", output_tensor->get_byte_size(), ") ",
                        "than required (", output_memory->size(), ")");

        bool need_output_update = output_layout.bytes_count() == 0 || (output_memory && output_tensor->get_byte_size() != output_memory->size());

        // Check shape is changed when size between output_tensor and output_memory are same.
        if (!need_output_update) {
            auto output_layout_shape = output_layout.get_shape();
            auto output_tensor_shape = output_tensor->get_shape();
            if (!output_layout_shape.empty() && !output_tensor_shape.empty()) {
                if (output_layout_shape != output_tensor_shape) {
                    need_output_update = true;
                }
            }
        }

        if (need_output_update) {
            OV_ITT_SCOPED_TASK(itt::domains::intel_gpu_plugin, "SyncInferRequest::wait::update_output");
            auto mem_shape = output_layout.get_shape();
            // In case of old shape infer we need to shrink out tensor shape to avoid redudnant dimensions that occur due to rank extension
            // For new shape infer this shouldn't happen, thus remove that WA once we migrate to ngraph-based shape infer for all cases
            if (!m_graph->get_config().get_property(ov::intel_gpu::allow_new_shape_infer)) {
                OPENVINO_ASSERT(port.get_partial_shape().is_static(), "[GPU] Unexpected dynamic shape for legacy shape inference");
                OPENVINO_ASSERT(ov::shape_size(port.get_shape()) == ov::shape_size(mem_shape), "[GPU] Unexpected elements count for output tensor");
                mem_shape = port.get_shape();
            }
            if (is_dynamic) {
                bool need_reallocate = true;
                auto usm_host_tensor = std::dynamic_pointer_cast<USMHostTensor>(output_tensor);
                if (usm_host_tensor && output_memory)
                    need_reallocate = usm_host_tensor->get_impl()->get_original_memory()->size() < output_memory->size();
                else if (!is_remote && output_memory)
                    need_reallocate = output_tensor_wrapper.actual_size < output_memory->size();

                if (need_reallocate) {
                    std::string internal_name = m_output_names_map.at(port_idx);
                    auto actual_memory_shape = predict_shape(internal_name, cldnn::layout(mem_shape,
                                                                                          output_tensor->get_element_type(),
                                                                                          cldnn::format::get_default_format(mem_shape.size())),
                                                             *m_shape_predictor);
                    output_tensor->set_shape(actual_memory_shape);
                }
            }

            output_tensor->set_shape(mem_shape);
        }

        // mapping remote blobs not needed -
        // let the user take care of them explicitly
        if (!is_remote && output_memory) {
            auto dst_ptr = static_cast<uint8_t*>(output_tensor->data());
            bool same_mem = same_host_mem(output_memory, dst_ptr);
            if (!same_mem && output_memory->size()) {
                GPU_DEBUG_TRACE_DETAIL << internal_name << " with index " << port_idx << " copy from: " << output_memory->buffer_ptr() << " to "
                                       << (!is_remote ? output_tensor->data() : remote_ptr->get_original_memory()->buffer_ptr()) << std::endl;
                if (auto ev = copy_output_data(output_memory, *output_tensor)) {
                    copy_events.push_back(ev);
                }
            }
        } else if (is_remote && is_dynamic) {
            auto& stream = m_graph->get_network()->get_stream();
            auto user_mem = remote_ptr->get_original_memory();
            if (user_mem->get_allocation_type() == cldnn::allocation_type::cl_mem && output_memory->get_allocation_type() != cldnn::allocation_type::cl_mem) {
                // WA: Copy between cl_mem and usm memory may fail for some reason (driver bug?)
                // so this explicit memcpy is used to provide correct output for cl_mem output in dynamic cases
                cldnn::mem_lock<uint8_t, cldnn::mem_lock_type::write> lock_dst(user_mem, stream);
                cldnn::mem_lock<uint8_t, cldnn::mem_lock_type::read> lock_src(output_memory, stream);
                std::memcpy(lock_dst.data(), lock_src.data(), output_memory->size());
            } else {
                copy_events.push_back(output_memory->copy_to(stream, *user_mem, false));
            }
        }
    }

    if (!copy_events.empty()) {
        auto& stream = network.get_stream();
        if (stream.get_queue_type() == QueueTypes::in_order) {
            // wait only the last one
            stream.wait_for_events({copy_events.back()});
        } else {
            stream.wait_for_events(copy_events);
        }
    }

    // finally collect profiling info
    if (m_enable_profiling) {
        m_graph->update_profiling_info();
    }
}

// ----------------------------------------------------------------------------------------- //
// ---------------------------- internal utils --------- ----------------------------------- //
// ----------------------------------------------------------------------------------------- //
void SyncInferRequest::setup_stream_graph() {
    int stream_id = 0;
    auto& stream_graphs = std::static_pointer_cast<const CompiledModel>(get_compiled_model())->get_graphs();
    if (nullptr != m_stream_executor) {
        stream_id = m_stream_executor->get_stream_id();
        auto num_graphs = stream_graphs.size();
        stream_id = stream_id % num_graphs;
    }
    m_graph = stream_graphs[stream_id];
}

std::shared_ptr<ov::ITensor> SyncInferRequest::create_host_tensor(const ov::PartialShape& port_shape, const ov::element::Type& port_element_type) const {
    OV_ITT_SCOPED_TASK(itt::domains::intel_gpu_plugin, "SyncInferRequest::create_host_tensor");
    return m_context->create_host_tensor(port_element_type, get_tensor_shape(port_shape))._ptr;
}

std::shared_ptr<ov::ITensor> SyncInferRequest::create_device_tensor(const ov::PartialShape& port_shape, ov::element::Type element_type,
                                                                    bool need_lockable_memory) const {
    TensorType tensor_type = TensorType::BT_EMPTY;
    if (m_graph->get_engine().use_unified_shared_memory()) {
        tensor_type = need_lockable_memory ? TensorType::BT_USM_HOST_INTERNAL : TensorType::BT_USM_DEVICE_INTERNAL;
    } else {
        tensor_type = TensorType::BT_BUF_INTERNAL;
    }

    // Create OpenCL buffer for PVC if lockable memory is needed due to performance issue with usm host
    if (!can_use_usm_host(m_graph->get_engine()) && need_lockable_memory)
        tensor_type = TensorType::BT_BUF_INTERNAL;

    return std::make_shared<RemoteTensorImpl>(m_context,
                                              get_tensor_shape(port_shape),
                                              cldnn::element_type_to_data_type(element_type),
                                              tensor_type);
}

TensorWrapper SyncInferRequest::create_or_share_device_tensor(const TensorWrapper& user_tensor_wrapper,
                                                              const std::string& name,
                                                              const ov::PartialShape& port_pshape,
                                                              ov::element::Type element_type,
                                                              bool need_lockable_mem) const {
    auto& engine = m_graph->get_engine();
    auto user_tensor = user_tensor_wrapper.ptr;
    auto tensor_shape = user_tensor->get_shape();
    bool is_dynamic = port_pshape.is_dynamic();
    OPENVINO_ASSERT(std::dynamic_pointer_cast<RemoteTensorImpl>(user_tensor) == nullptr, "[GPU] Unexpected remote tensor");
    auto usm_host_tensor = std::dynamic_pointer_cast<USMHostTensor>(user_tensor);

    // Note: currently, using USM Host memory for dGPUs in some scenarios (LLMs) leads to performance degradation,
    // so apply wider USM Host memory type detection only for iGPUs
    auto user_tensor_mem_type = engine.detect_usm_allocation_type(user_tensor->data());
    auto usm_host_raw_ptr = engine.get_device_info().dev_type == cldnn::device_type::integrated_gpu &&
                            user_tensor_mem_type == cldnn::allocation_type::usm_host;

    bool can_share = !is_convert_required(user_tensor->get_element_type(), element_type) && can_use_usm_host(engine);

    if (usm_host_tensor && can_share) {
        return { usm_host_tensor->get_impl(), user_tensor_wrapper.owner };
    } else if (usm_host_raw_ptr && can_share) {
        return { std::make_shared<RemoteTensorImpl>(m_context,
                                                    user_tensor->get_shape(),
                                                    cldnn::element_type_to_data_type(element_type),
                                                    TensorType::BT_USM_SHARED,
                                                    user_tensor->data()), TensorOwner::USER };
    }

    auto actual_memory_shape = tensor_shape;
    if (is_dynamic) {
        actual_memory_shape = predict_shape(name, cldnn::layout(tensor_shape,
                                                                element_type,
                                                                cldnn::format::get_default_format(tensor_shape.size())),
                                            *m_shape_predictor);
    }

    return { create_device_tensor(actual_memory_shape, element_type, need_lockable_mem), TensorOwner::PLUGIN };
}

cldnn::event::ptr SyncInferRequest::copy_output_data(cldnn::memory::ptr src, const ov::ITensor& dst) const {
    OV_ITT_SCOPED_TASK(itt::domains::intel_gpu_plugin, "SyncInferRequest::copy_output_data");
    OPENVINO_ASSERT(src->count() <= dst.get_size(),
                    "[GPU] Unexpected elements count of dst tensor: ",
                    "expected at least ", src->count(), ", but ",
                    "only ", dst.get_size(), " got");

    const auto& layout = src->get_layout();
    auto& stream = m_graph->get_network()->get_stream();

    if (is_convert_required(layout.data_type, dst.get_element_type())) {
        convert_and_copy(src, &dst, stream);
        return nullptr;
    } else {
        return src->copy_to(stream, dst.data(), false);
    }
}

void SyncInferRequest::allocate_input(const ov::Output<const ov::Node>& port, size_t input_idx) {
    const auto& shape = port.get_partial_shape();
    auto element_type = port.get_element_type();

    m_user_inputs[input_idx] = { create_host_tensor(shape, element_type), TensorOwner::PLUGIN };
    ov::ISyncInferRequest::set_tensor(port, m_user_inputs.at(input_idx).ptr);
}

void SyncInferRequest::allocate_output(const ov::Output<const ov::Node>& port, size_t output_idx) {
    const auto& shape = port.get_partial_shape();
    auto element_type = port.get_element_type();

    m_user_outputs[output_idx] = { create_host_tensor(shape, element_type), TensorOwner::PLUGIN };
    ov::ISyncInferRequest::set_tensor(port, m_user_outputs.at(output_idx).ptr);
}

void SyncInferRequest::allocate_inputs() {
    OV_ITT_SCOPED_TASK(itt::domains::intel_gpu_plugin, "SyncInferRequest::allocate_inputs");

    for (const auto& it : m_input_ports_map) {
        size_t input_idx = it.first;
        const auto& port = it.second;
        GPU_DEBUG_LOG << "[init input blob with index: " << input_idx << "]" << std::endl;

        bool is_nv12_input = false;
        if (port.get_rt_info().count(ov::preprocess::TensorInfoMemoryType::get_type_info_static())) {
            std::string mem_type = port.get_rt_info().at(ov::preprocess::TensorInfoMemoryType::get_type_info_static())
                                                     .as<ov::preprocess::TensorInfoMemoryType>().value;
            if (mem_type.find(ov::intel_gpu::memory_type::surface) != std::string::npos) {
                is_nv12_input = true;
            }
        }

        if (!is_nv12_input) {
            allocate_input(port, input_idx);
        }
    }
}

void SyncInferRequest::allocate_outputs() {
    OV_ITT_SCOPED_TASK(itt::domains::intel_gpu_plugin, "SyncInferRequest::allocate_outputs");

    // allocate outputs
    for (const auto& it : m_output_ports_map) {
        size_t output_idx = it.first;
        const auto& port = it.second;
        GPU_DEBUG_LOG << "[init output blob with index: " << output_idx << "]" << std::endl;

        allocate_output(port, output_idx);
    }
}

void SyncInferRequest::allocate_states() {
    const auto& network = m_graph->get_network();
    const auto& variables_info = network->get_variables_info();
    for (auto& vi : variables_info) {
        const auto& state_prims = vi.second.m_primitives;
        bool indirect_kv_cache = false;
        int64_t beam_axis = 0;
        int64_t concat_axis = 0;
        auto kv_cache_shape = vi.second.m_layout.get_partial_shape();
        for (auto& p : state_prims) {
            if (auto kv_cache_prim = dynamic_cast<const cldnn::kv_cache*>(p)) {
                indirect_kv_cache = kv_cache_prim->indirect;
                beam_axis = ov::util::normalize(kv_cache_prim->gather_axis, kv_cache_shape.size());
                concat_axis = ov::util::normalize(kv_cache_prim->concat_axis, kv_cache_shape.size());
            }
        }

        if (indirect_kv_cache) {
            m_variables.emplace(vi.first, std::make_shared<VariableStateIndirectKVCache>(vi.second, m_context, m_shape_predictor, beam_axis, concat_axis));
        } else {
            m_variables.emplace(vi.first, std::make_shared<VariableState>(vi.second, m_context, m_shape_predictor));
        }
    }
}

void SyncInferRequest::prepare_state(const std::string& name, const std::shared_ptr<VariableStateBase>& variable) {
    m_graph->get_network()->set_variable(name, variable);
}

std::vector<cldnn::event::ptr> SyncInferRequest::prepare_batched_input(size_t input_idx,
                                                                       const ov::Output<const ov::Node>& port,
                                                                       const std::vector<ov::SoPtr<ov::ITensor>>& user_tensors) {
    std::vector<cldnn::event::ptr> ret_events;
    bool is_host = all_host_tensors(user_tensors);
    bool is_remote_buffer = all_remote_buffers(user_tensors);
    const cldnn::primitive::primitive_id_arr& internal_names = m_graph->input_port_index_to_internal(input_idx);
    // Host buffers are merged to single tensor
    if (is_host || is_remote_buffer) {
        auto tmp_shape = user_tensors.at(0)->get_shape();
        auto tmp_et = user_tensors.at(0)->get_element_type();
        tmp_shape[0] = user_tensors.size();
        std::shared_ptr<ov::ITensor> merged_tensor = nullptr;
        if (is_host) {
            merged_tensor = m_context->create_host_tensor(tmp_et, tmp_shape)._ptr;
            auto ptr = static_cast<uint8_t*>(merged_tensor->data());
            ov::parallel_for(user_tensors.size(), [&](size_t i) {
                const auto& tensor = user_tensors.at(i);
                std::memcpy(ptr + i * tensor->get_byte_size(), static_cast<uint8_t*>(tensor->data()), tensor->get_byte_size());
            });
        } else {
            const auto& stream = m_graph->get_network()->get_stream();
            merged_tensor = m_context->create_tensor(tmp_et, tmp_shape, {})._ptr;
            auto merged_memory = std::dynamic_pointer_cast<RemoteTensorImpl>(merged_tensor)->get_memory();
            cldnn::mem_lock<uint8_t> dst_lock(merged_memory, stream);
            for (size_t i = 0; i < user_tensors.size(); i++) {
                auto input_tensor = std::dynamic_pointer_cast<RemoteTensorImpl>(user_tensors[i]._ptr);
                cldnn::mem_lock<uint8_t> src_lock(input_tensor->get_memory(), stream);
                std::memcpy(dst_lock.data() + i * input_tensor->get_byte_size(), src_lock.data(), input_tensor->get_byte_size());
            }
        }

        auto events = prepare_input(internal_names[0], input_idx, port, {merged_tensor, TensorOwner::PLUGIN});
        std::move(events.begin(), events.end(), std::back_inserter(ret_events));
    } else {
        OPENVINO_ASSERT(user_tensors.size() == internal_names.size(), "[GPU] Internal names and user tensors size mismatch");
        for (size_t i = 0; i < user_tensors.size(); i++) {
            auto events = prepare_input(internal_names[i], input_idx, port, {user_tensors[i]._ptr, TensorOwner::USER});
            std::move(events.begin(), events.end(), std::back_inserter(ret_events));
        }
    }

    return ret_events;
}

std::vector<cldnn::event::ptr> SyncInferRequest::prepare_input(const std::string& internal_name,
                                                               size_t input_idx,
                                                               const ov::Output<const ov::Node>& port,
                                                               const TensorWrapper& user_tensor_wrapper) {
    OV_ITT_SCOPED_TASK(itt::domains::intel_gpu_plugin, openvino::itt::handle("SyncInferRequest::prepare_input: " + internal_name));
    auto pshape = port.get_partial_shape();
    auto is_dynamic = pshape.is_dynamic();
    auto user_tensor = user_tensor_wrapper.ptr;
    auto element_type = user_tensor->get_element_type();

    auto remote_ptr = std::dynamic_pointer_cast<RemoteTensorImpl>(user_tensor);
    auto usm_host_ptr = std::dynamic_pointer_cast<USMHostTensor>(user_tensor);
    bool is_remote = remote_ptr != nullptr;
    bool is_usm_host_tensor = usm_host_ptr != nullptr;

    GPU_DEBUG_TRACE_DETAIL << "Prepare input for " << internal_name
            << " ( is_remote ? " << is_remote << ", is_usm_host_tensor ? " << is_usm_host_tensor << ")" << std::endl;
    GPU_DEBUG_TRACE_DETAIL << "    port shape       : " << pshape.to_string() << std::endl;
    GPU_DEBUG_TRACE_DETAIL << "    user_tensor shape: " << user_tensor->get_shape().to_string() << std::endl;

    auto network = m_graph->get_network();
    auto& engine = m_graph->get_engine();
    auto& stream = network->get_stream();

    OPENVINO_ASSERT(pshape.compatible(ov::PartialShape(user_tensor->get_shape())) || is_batched_input(port),
                    "[GPU] The input tensor size is not equal to model port shape, can't handle input tensor with name: ",
                    internal_name,
                    ", because model input (shape=",
                    pshape,
                    ") and tensor (shape=",
                    user_tensor->get_shape(),
                    ") are incompatible");

    auto device_tensor_et = convert_to_supported_device_type(element_type);
    bool convert_needed = is_convert_required(element_type, device_tensor_et);

    auto print_arr = [&](int64_t* vec, size_t max_len) {
        std::stringstream ss;
        for (size_t i = 0; i < max_len; i++) {
            ss << vec[i] << ", ";
        }
        GPU_DEBUG_TRACE_DETAIL << "Set_input parameter:input_ids (max-len=" << max_len << ") content: " << ss.str() << "\n";
    };

    if (internal_name == "parameter:input_ids") {
        print_arr(static_cast<int64_t*>(user_tensor->data()), user_tensor->get_size());
    }

    if (is_remote) {
        if (convert_needed) {
            m_plugin_inputs[input_idx] = { create_device_tensor(pshape,
                                                                cldnn::element_type_to_data_type(element_type),
                                                                false), TensorOwner::PLUGIN };
        } else {
            m_plugin_inputs[input_idx] = user_tensor_wrapper;
        }
    } else if (is_usm_host_tensor && !convert_needed && can_use_usm_host(engine)) {
        if (element_type != cldnn::element_type_to_data_type(element_type)) {
            m_plugin_inputs[input_idx] = { std::make_shared<RemoteTensorImpl>(m_context,
                                                                              user_tensor->get_shape(),
                                                                              cldnn::element_type_to_data_type(element_type),
                                                                              TensorType::BT_USM_SHARED,
                                                                              user_tensor->data()), TensorOwner::USER };
        } else {
            m_plugin_inputs[input_idx] = { usm_host_ptr->get_impl(), user_tensor_wrapper.owner };
        }
        is_remote = true;
    }

    auto user_tensor_mem_type = cldnn::allocation_type::unknown;
    if (!is_remote) {
        user_tensor_mem_type = engine.detect_usm_allocation_type(user_tensor_wrapper.ptr->data());
    }

    auto plugin_tensor_mem_type = cldnn::allocation_type::unknown;
    if (m_plugin_inputs.count(input_idx)) {
        plugin_tensor_mem_type = std::dynamic_pointer_cast<RemoteTensorImpl>(m_plugin_inputs[input_idx].ptr)->get_original_memory()->get_allocation_type();
    }

    // Note: currently, using USM Host memory for dGPUs in some scenarios (LLMs) leads to performance degradation,
    // so apply wider USM Host memory type detection only for iGPUs
    auto usm_host_raw_ptr = engine.get_device_info().dev_type == cldnn::device_type::integrated_gpu &&
                            user_tensor_mem_type == cldnn::allocation_type::usm_host;

    bool update_device_tensor = (m_plugin_inputs.count(input_idx) == 0) ||
                                (m_plugin_inputs[input_idx].owner == TensorOwner::USER && !is_remote) ||
                                (plugin_tensor_mem_type != cldnn::allocation_type::usm_host && usm_host_raw_ptr);

    if (update_device_tensor) {
        // If device input hasn't been created, then try to use user memory if it's usm_host, or allocate new device buffer
        m_plugin_inputs[input_idx] = create_or_share_device_tensor(user_tensor_wrapper, internal_name, pshape, device_tensor_et, convert_needed);
    } else if (!is_remote) {
        // Device memory has been created on previous iterations. Try to reuse whenever it's possible
        auto device_tensor_wrapper = m_plugin_inputs.at(input_idx);
        auto device_tensor = std::dynamic_pointer_cast<RemoteTensorImpl>(device_tensor_wrapper.ptr);
        if (is_dynamic) {
            if (device_tensor->get_original_memory()->size() < user_tensor->get_byte_size()) {
                auto actual_shape = predict_shape(internal_name, cldnn::layout(user_tensor->get_shape(),
                                                                               element_type,
                                                                               cldnn::format::get_default_format(user_tensor->get_shape().size())),
                                                  *m_shape_predictor);
                GPU_DEBUG_TRACE_DETAIL << "    actual memory shape: " << actual_shape.to_string() << std::endl;
                auto new_tensor = create_device_tensor(actual_shape, device_tensor_et, false);
                new_tensor->set_shape(user_tensor->get_shape());
                m_plugin_inputs[input_idx] = { new_tensor, TensorOwner::PLUGIN };
            }
        }
    }

    auto device_tensor = std::dynamic_pointer_cast<RemoteTensorImpl>(m_plugin_inputs.at(input_idx).ptr);
    if (is_dynamic) {
        OPENVINO_ASSERT(device_tensor->get_original_memory()->size() >= user_tensor->get_size(),
                        "[GPU] Size of input device tensor (=",
                        device_tensor->get_original_memory()->size(),
                        ") is expected to be greater or equal to user tensor (=",
                        user_tensor->get_size(),
                        ") in dynamic case for ", internal_name);
        // tensor reshape below is expected to work w/o reallocation
        device_tensor->set_shape(user_tensor->get_shape());
    } else {
        OPENVINO_ASSERT(device_tensor->get_size() == user_tensor->get_size(),
                        "[GPU] Size of user tensor (=",
                        user_tensor->get_size(),
                        ") and device tensor (=",
                        device_tensor->get_size(),
                        ") don't match for ", internal_name,
                        ". Those are expected to be equal in case of static shape of the port");
    }

    auto memory = device_tensor->get_memory();
    // WA to extend shape to ranks expected by legacy shape infer. Remove after full migration to new shape infer
    if (!m_graph->get_config().get_property(ov::intel_gpu::allow_new_shape_infer)) {
        auto new_layout = memory->get_layout();
        new_layout.set_partial_shape(m_graph->get_input_layouts().at(input_idx).get_shape());
        memory = engine.reinterpret_buffer(*memory, new_layout);
    }

    cldnn::event::ptr ret_event = nullptr;
    if (!is_remote && !convert_needed) {
        auto src_ptr = static_cast<uint8_t*>(user_tensor->data());
        if (!same_host_mem(memory, src_ptr)) {
            ret_event = memory->copy_from(stream, src_ptr, false);
        }
    }
    if (convert_needed) {
        if (is_remote) {
            convert_and_copy(remote_ptr->get_memory(), device_tensor->get_memory(), stream);
        } else {
            convert_and_copy(user_tensor.get(), device_tensor.get(), stream);
        }
    }

    GPU_DEBUG_TRACE_DETAIL << internal_name << " with index " << input_idx << " prepare input: " << memory->buffer_ptr()
                           << " alloc_type: " << memory->get_allocation_type() << std::endl;
    network->set_input_data(internal_name, memory);

    if (ret_event && !ret_event->is_set())
        return { ret_event };
    else
        return {};
}

std::vector<cldnn::event::ptr> SyncInferRequest::prepare_output(size_t output_idx,
                                                                const ov::Output<const ov::Node>& port,
                                                                const TensorWrapper& user_tensor_wrapper) {
    OV_ITT_SCOPED_TASK(itt::domains::intel_gpu_plugin, "SyncInferRequest::prepare_output");
    auto pshape = port.get_partial_shape();
    auto is_dynamic = pshape.is_dynamic();
    auto element_type = port.get_element_type();
    auto user_tensor = user_tensor_wrapper.ptr;
    auto remote_ptr = std::dynamic_pointer_cast<RemoteTensorImpl>(user_tensor);
    auto internal_name = m_output_names_map.at(output_idx);
    bool is_remote = remote_ptr != nullptr;

    GPU_DEBUG_TRACE_DETAIL << "Prepare output for " << internal_name << std::endl;
    GPU_DEBUG_TRACE_DETAIL << "    port shape       : " << pshape.to_string() << std::endl;
    GPU_DEBUG_TRACE_DETAIL << "    user_tensor shape: " << user_tensor->get_shape().to_string() << std::endl;

    if (user_tensor->get_size() > 0) {
        OPENVINO_ASSERT(pshape.compatible(ov::PartialShape(user_tensor->get_shape())),
                        "[GPU] The output tensor size is not equal to model port shape, can't handle output tensor with name: ",
                        internal_name,
                        ", because model output (shape=",
                        pshape,
                        ") and tensor (shape=",
                        user_tensor->get_shape(),
                        ") are incompatible");
    }

    auto network = m_graph->get_network();
    auto device_tensor_et = convert_to_supported_device_type(element_type);
    bool convert_needed = is_convert_required(device_tensor_et, element_type);

    if (is_remote && !convert_needed && !is_dynamic) {
        m_plugin_outputs[output_idx] = user_tensor_wrapper;
    }

    if (!is_dynamic) {
        auto is_cpu_impl = network->is_cpu_impl(internal_name);
        bool has_device_buffer = m_plugin_outputs.count(output_idx) > 0;
        bool update_device_tensor = !has_device_buffer ||
                                    (m_plugin_outputs[output_idx].owner == TensorOwner::USER && !is_remote);
        if (update_device_tensor) {
            m_plugin_outputs[output_idx] =
                create_or_share_device_tensor(user_tensor_wrapper, internal_name, pshape, device_tensor_et, is_cpu_impl || convert_needed);
        }
    }

    // Missing output in _plugin_outputs means that the network is dynamic and outputs couldn't be pre-allocated
    if (m_plugin_outputs.find(output_idx) == m_plugin_outputs.end())
        return {};

    auto output_tensor = std::dynamic_pointer_cast<RemoteTensorImpl>(m_plugin_outputs.at(output_idx).ptr);
    auto output_memory = output_tensor->get_memory();
    GPU_DEBUG_TRACE_DETAIL << internal_name << " with index " << output_idx << " prepare output: " << output_memory->buffer_ptr() << std::endl;
    return network->set_output_memory(internal_name, output_memory);
}

void SyncInferRequest::init_mappings() {
    const auto& inputs = get_inputs();
    for (size_t input_idx = 0; input_idx < inputs.size(); ++input_idx) {
        m_input_ports_map[input_idx] = inputs[input_idx];
    }

    const auto& outputs = get_outputs();
    for (size_t output_idx = 0; output_idx < outputs.size(); ++output_idx) {
        m_output_ports_map[output_idx] = outputs[output_idx];
        m_output_names_map[output_idx] = m_graph->out_port_index_to_internal(output_idx);
    }
}

bool SyncInferRequest::is_batched_input(const ov::Output<const ov::Node>& port) const {
    return m_batched_tensors.count(port.get_tensor_ptr()) > 0;
}

}  // namespace intel_gpu
}  // namespace ov
