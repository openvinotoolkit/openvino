// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "failsafe.hpp"

#include <cstring>
#include <exception>
#include <utility>

#include "../../logging.hpp"
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

}  // namespace

ov::SoPtr<ov::ICompiledModel> ov::npuw::failsafe::CompiledModel::create(
    const std::shared_ptr<ov::Model>& model,
    const std::shared_ptr<const ov::IPlugin>& plugin,
    const std::vector<std::string>& devices,
    const Factory& factory) {
    OPENVINO_ASSERT(!devices.empty(), "Failsafe compiled model requires at least one device");
    OPENVINO_ASSERT(static_cast<bool>(factory), "Failsafe compiled model requires a factory");

    if (devices.size() == 1u) {
        auto compiled_model = factory(devices.front());
        OPENVINO_ASSERT(compiled_model._ptr != nullptr,
                        "Failsafe factory returned null compiled model for device ",
                        devices.front());
        return compiled_model;
    }

    auto compiled_model = std::make_shared<CompiledModel>(model, plugin, devices, factory);
    std::lock_guard<std::mutex> lock(compiled_model->m_mutex);
    compiled_model->ensure_active_compiled_model_locked();
    return {compiled_model, {}};
}

ov::npuw::failsafe::CompiledModel::CompiledModel(const std::shared_ptr<ov::Model>& model,
                                                 const std::shared_ptr<const ov::IPlugin>& plugin,
                                                 const std::vector<std::string>& devices,
                                                 const Factory& factory)
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
            OPENVINO_ASSERT(compiled_model._ptr != nullptr,
                            "Failsafe factory returned null compiled model for device ",
                            m_devices[idx]);
            m_active_state = ActiveState{idx, 0u, std::move(compiled_model)};
            return m_active_state.value();
        } catch (...) {
            last_failure = std::current_exception();
            LOG_WARN("Failed to compile for "
                     << m_devices[idx] << (idx + 1 < m_devices.size() ? "; trying the next fallback device." : "."));
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
            OPENVINO_ASSERT(compiled_model._ptr != nullptr,
                            "Failsafe factory returned null compiled model for device ",
                            m_devices[idx]);
            m_active_state = ActiveState{idx, current.generation + 1u, std::move(compiled_model)};
            return m_active_state.value();
        } catch (...) {
            failure = std::current_exception();
            LOG_WARN("Failed to compile for " << m_devices[idx] << " during failover.");
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

ov::Output<const ov::Node> ov::npuw::failsafe::InferRequest::inner_port_locked(
    const ov::Output<const ov::Node>& port) const {
    auto found = find_port(port);
    OPENVINO_ASSERT(found.found(), "Cannot find failsafe port ", port);
    // Device models may expose cloned nodes, so use the public I/O position.
    const auto& model = m_request->get_compiled_model();
    const auto& ports = found.is_output() ? model->outputs() : model->inputs();
    OPENVINO_ASSERT(found.idx < ports.size(), "Failsafe device model has incompatible ports");
    return ports[found.idx];
}

void ov::npuw::failsafe::InferRequest::infer() {
    std::lock_guard<std::mutex> lock(m_mutex);

    std::vector<ov::SoPtr<ov::ITensor>> input_tensors;
    std::vector<ov::SoPtr<ov::ITensor>> output_tensors;
    bool need_rebind = false;

    ensure_inner_request_locked();
    for (const auto& port : m_request->get_compiled_model()->inputs()) {
        input_tensors.push_back(m_request->get_tensor(port));
    }
    for (const auto& port : m_request->get_compiled_model()->outputs()) {
        output_tensors.push_back(m_request->get_tensor(port));
    }

    while (true) {
        ensure_inner_request_locked();
        try {
            if (need_rebind) {
                const auto& inputs = m_request->get_compiled_model()->inputs();
                const auto& outputs = m_request->get_compiled_model()->outputs();
                OPENVINO_ASSERT(inputs.size() == input_tensors.size() && outputs.size() == output_tensors.size(),
                                "Failsafe device model has incompatible port counts");
                for (size_t i = 0; i < inputs.size(); ++i) {
                    m_request->set_tensor(inputs[i], input_tensors[i]);
                }
                for (size_t i = 0; i < outputs.size(); ++i) {
                    m_request->set_tensor(outputs[i], output_tensors[i]);
                }
            }
            m_request->infer();
            need_rebind = false;
            return;
        } catch (...) {
            LOG_WARN("Subgraph inference failed; attempting a fallback device.");
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
    return m_request->get_tensor(inner_port_locked(port));
}

void ov::npuw::failsafe::InferRequest::set_tensor(const ov::Output<const ov::Node>& port,
                                                  const ov::SoPtr<ov::ITensor>& tensor) {
    std::lock_guard<std::mutex> lock(m_mutex);
    ensure_inner_request_locked();
    m_request->set_tensor(inner_port_locked(port), tensor);
}

void ov::npuw::failsafe::InferRequest::check_tensors() const {}

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
