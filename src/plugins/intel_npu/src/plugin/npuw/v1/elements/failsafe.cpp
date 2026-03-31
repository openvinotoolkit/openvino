// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "failsafe.hpp"

#include <cstring>
#include <exception>
#include <utility>

#include "openvino/runtime/make_tensor.hpp"

namespace {

[[noreturn]] void throw_failsafe_error(const char* stage, const std::exception_ptr& failure) {
    try {
        if (failure) {
            std::rethrow_exception(failure);
        }
    } catch (const std::exception& ex) {
        OPENVINO_THROW("Failsafe ", stage, " fallback exhausted: ", ex.what());
    } catch (...) {
        OPENVINO_THROW("Failsafe ", stage, " fallback exhausted");
    }

    OPENVINO_THROW("Failsafe ", stage, " fallback exhausted");
}

ov::SoPtr<ov::ITensor> allocate_tensor_like(const ov::SoPtr<ov::ITensor>& tensor) {
    return ov::get_tensor_impl(ov::Tensor(tensor->get_element_type(), tensor->get_shape()));
}

// The wrapper keeps public output tensors stable across failover. For plain
// host tensors the generic ITensor::copy_to path is not consistently available
// in lightweight test doubles, so use memcpy for the common contiguous case and
// fall back to plugin-provided copy_to for non-contiguous / remote tensors.
void copy_tensor_data(const ov::SoPtr<ov::ITensor>& src, const ov::SoPtr<ov::ITensor>& dst) {
    OPENVINO_ASSERT(src->get_byte_size() == dst->get_byte_size(), "Failsafe tensor copy size mismatch");
    if (src->is_continuous() && dst->is_continuous()) {
        std::memcpy(dst->data(), src->data(), src->get_byte_size());
        return;
    }
    src->copy_to(dst._ptr);
}

}  // namespace

std::shared_ptr<ov::ICompiledModel> ov::npuw::failsafe::CompiledModel::create(
    const std::shared_ptr<ov::Model>& model,
    const std::shared_ptr<const ov::IPlugin>& plugin,
    const std::vector<std::string> &devices,
    const Factory &factory) {
    OPENVINO_ASSERT(!devices.empty(), "Failsafe compiled model requires at least one device");
    OPENVINO_ASSERT(static_cast<bool>(factory), "Failsafe compiled model requires a factory");

    if (devices.size() == 1u) {
        auto compiled_model = factory(devices.front());
        OPENVINO_ASSERT(compiled_model != nullptr,
                        "Failsafe factory returned null compiled model for device ",
                        devices.front());
        return compiled_model;
    }

    auto compiled_model = std::make_shared<CompiledModel>(model, plugin, devices, factory);
    std::lock_guard<std::mutex> lock(compiled_model->m_mutex);
    compiled_model->ensure_active_compiled_model_locked();
    return compiled_model;
}

ov::npuw::failsafe::CompiledModel::CompiledModel(const std::shared_ptr<ov::Model>& model,
                                                 const std::shared_ptr<const ov::IPlugin>& plugin,
                                                 const std::vector<std::string> &devices,
                                                 const Factory &factory)
     : ov::ICompiledModel(model, plugin),
       m_devices(devices),
       m_factory(factory) {
    OPENVINO_ASSERT(!m_devices.empty(), "Failsafe compiled model requires at least one device");
    OPENVINO_ASSERT(static_cast<bool>(m_factory), "Failsafe compiled model requires a factory");
}

ov::npuw::failsafe::CompiledModel::ActiveState ov::npuw::failsafe::CompiledModel::ensure_active_compiled_model_locked()
    const {
    if (m_active_state.has_value()) {
        return m_active_state.value();
    }

    std::exception_ptr last_failure;
    for (std::size_t idx = 0; idx < m_devices.size(); ++idx) {
        try {
            auto compiled_model = m_factory(m_devices[idx]);
            OPENVINO_ASSERT(compiled_model != nullptr,
                            "Failsafe factory returned null compiled model for device ",
                            m_devices[idx]);
            m_active_state = ActiveState{idx, 0u, std::move(compiled_model)};
            return m_active_state.value();
        } catch (...) {
            last_failure = std::current_exception();
        }
    }

    throw_failsafe_error("compile", last_failure);
}

ov::npuw::failsafe::CompiledModel::ActiveState ov::npuw::failsafe::CompiledModel::failover_from_locked(
    std::size_t generation,
    const char* stage,
    std::exception_ptr failure) const {
    auto current = ensure_active_compiled_model_locked();
    if (current.generation != generation) {
        return current;
    }

    for (std::size_t idx = current.device_index + 1; idx < m_devices.size(); ++idx) {
        try {
            auto compiled_model = m_factory(m_devices[idx]);
            OPENVINO_ASSERT(compiled_model != nullptr,
                            "Failsafe factory returned null compiled model for device ",
                            m_devices[idx]);
            m_active_state = ActiveState{idx, current.generation + 1u, std::move(compiled_model)};
            return m_active_state.value();
        } catch (...) {
            failure = std::current_exception();
        }
    }

    throw_failsafe_error(stage, failure);
}

std::shared_ptr<ov::IAsyncInferRequest> ov::npuw::failsafe::CompiledModel::create_request(
    std::size_t& generation) const {
    std::lock_guard<std::mutex> lock(m_mutex);

    while (true) {
        const auto current = ensure_active_compiled_model_locked();
        try {
            auto request = current.compiled_model->create_infer_request();
            OPENVINO_ASSERT(request != nullptr,
                            "Failsafe compiled model returned null infer request for device ",
                            m_devices[current.device_index]);
            generation = current.generation;
            return request;
        } catch (...) {
            failover_from_locked(current.generation, "infer-request creation", std::current_exception());
        }
    }
}

bool ov::npuw::failsafe::CompiledModel::is_generation_current(std::size_t generation) const {
    std::lock_guard<std::mutex> lock(m_mutex);
    return ensure_active_compiled_model_locked().generation == generation;
}

std::size_t ov::npuw::failsafe::CompiledModel::active_device_index() const {
    std::lock_guard<std::mutex> lock(m_mutex);
    return ensure_active_compiled_model_locked().device_index;
}

std::string ov::npuw::failsafe::CompiledModel::active_device_name() const {
    std::lock_guard<std::mutex> lock(m_mutex);
    const auto current = ensure_active_compiled_model_locked();
    return m_devices[current.device_index];
}

bool ov::npuw::failsafe::CompiledModel::is_at_last_device() const {
    return active_device_index() + 1 == m_devices.size();
}

void ov::npuw::failsafe::CompiledModel::export_model(std::ostream& model) const {
    std::lock_guard<std::mutex> lock(m_mutex);
    ensure_active_compiled_model_locked().compiled_model->export_model(model);
}

std::shared_ptr<const ov::Model> ov::npuw::failsafe::CompiledModel::get_runtime_model() const {
    std::lock_guard<std::mutex> lock(m_mutex);
    return ensure_active_compiled_model_locked().compiled_model->get_runtime_model();
}

void ov::npuw::failsafe::CompiledModel::set_property(const ov::AnyMap& properties) {
    std::lock_guard<std::mutex> lock(m_mutex);
    ensure_active_compiled_model_locked().compiled_model->set_property(properties);
}

ov::Any ov::npuw::failsafe::CompiledModel::get_property(const std::string& name) const {
    std::lock_guard<std::mutex> lock(m_mutex);
    return ensure_active_compiled_model_locked().compiled_model->get_property(name);
}

std::shared_ptr<ov::ISyncInferRequest> ov::npuw::failsafe::CompiledModel::create_sync_infer_request() const {
    auto self = std::static_pointer_cast<const CompiledModel>(shared_from_this());
    auto request = std::make_shared<InferRequest>(std::move(self));
    request->ensure_inner_request_locked();
    return request;
}

std::shared_ptr<ov::IAsyncInferRequest> ov::npuw::failsafe::CompiledModel::create_infer_request() const {
    return std::make_shared<ov::IAsyncInferRequest>(create_sync_infer_request(),
                                                    get_task_executor(),
                                                    get_callback_executor());
}

ov::npuw::failsafe::InferRequest::InferRequest(std::shared_ptr<const CompiledModel> compiled_model)
    : ov::ISyncInferRequest(compiled_model),
      m_failsafe_compiled_model(std::move(compiled_model)) {}

void ov::npuw::failsafe::InferRequest::ensure_inner_request_locked() const {
    if (m_request != nullptr && m_failsafe_compiled_model->is_generation_current(m_generation)) {
        return;
    }

    m_request = m_failsafe_compiled_model->create_request(m_generation);
}

ov::npuw::failsafe::InferRequest::PortKey ov::npuw::failsafe::InferRequest::port_key_locked(
    const ov::Output<const ov::Node>& port) const {
    auto found = find_port(port);
    OPENVINO_ASSERT(found.found(), "Cannot find failsafe port ", port);
    return PortKey{port.get_node(), port.get_index(), found.is_output()};
}

bool ov::npuw::failsafe::InferRequest::is_output_port_locked(const ov::Output<const ov::Node>& port) const {
    return port_key_locked(port).is_output;
}


void ov::npuw::failsafe::InferRequest::infer() {
    std::lock_guard<std::mutex> lock(m_mutex);

    using Binding = std::pair<ov::Output<const ov::Node>, ov::SoPtr<ov::ITensor>>;
    std::vector<Binding> input_tensors;
    std::vector<Binding> output_tensors;
    bool need_rebind = false;

    ensure_inner_request_locked();
    for (auto &&port : m_failsafe_compiled_model->inputs()) {
        input_tensors.push_back(Binding{port, m_request->get_tensor(port)});
    }
    for (auto &&port : m_failsafe_compiled_model->outputs()) {
        output_tensors.push_back(Binding{port, m_request->get_tensor(port)});
    }

    while (true) {
        ensure_inner_request_locked();
        try {
            if (need_rebind) {
                for (auto &&binding : input_tensors) {
                    m_request->set_tensor(binding.first, binding.second);
                }
                for (auto &&binding : output_tensors) {
                    m_request->set_tensor(binding.first, binding.second);
                }
            }
            m_request->infer();
            need_rebind = false;
            return;
        } catch (...) {
            {
                std::lock_guard<std::mutex> model_lock(m_failsafe_compiled_model->m_mutex);
                m_failsafe_compiled_model->failover_from_locked(m_generation, "infer", std::current_exception());
            }
            m_request.reset();
            m_generation = std::numeric_limits<std::size_t>::max();
            need_rebind = true;
        }
    }
}

ov::SoPtr<ov::ITensor> ov::npuw::failsafe::InferRequest::get_tensor(const ov::Output<const ov::Node>& port) const {
    std::lock_guard<std::mutex> lock(m_mutex);
    ensure_inner_request_locked();
    return m_request->get_tensor(port);
}

void ov::npuw::failsafe::InferRequest::set_tensor(const ov::Output<const ov::Node>& port,
                                                  const ov::SoPtr<ov::ITensor>& tensor) {
    std::lock_guard<std::mutex> lock(m_mutex);
    ensure_inner_request_locked();
    m_request->set_tensor(port, tensor);
}

void ov::npuw::failsafe::InferRequest::check_tensors() const {
}

std::vector<ov::SoPtr<ov::IVariableState>> ov::npuw::failsafe::InferRequest::query_state() const {
    std::lock_guard<std::mutex> lock(m_mutex);
    ensure_inner_request_locked();
    return m_request->query_state();
}

std::vector<ov::ProfilingInfo> ov::npuw::failsafe::InferRequest::get_profiling_info() const {
    std::lock_guard<std::mutex> lock(m_mutex);
    ensure_inner_request_locked();
    return m_request->get_profiling_info();
}
