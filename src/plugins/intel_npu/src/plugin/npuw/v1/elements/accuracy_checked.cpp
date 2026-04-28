// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "accuracy_checked.hpp"

#include <utility>

#include "openvino/core/except.hpp"
#include "openvino/runtime/make_tensor.hpp"

namespace ov::npuw::accuracy_checked {

// ============================================================================
// CompiledModel
// ============================================================================

ov::SoPtr<ov::ICompiledModel> CompiledModel::create(const std::shared_ptr<ov::Model>& model,
                                                    const std::shared_ptr<const ov::IPlugin>& plugin,
                                                    ov::SoPtr<ov::ICompiledModel> main_compiled,
                                                    ov::SoPtr<ov::ICompiledModel> ref_compiled,
                                                    Checker checker) {
    OPENVINO_ASSERT(main_compiled._ptr != nullptr, "AccuracyChecked: main compiled model must not be null");
    OPENVINO_ASSERT(static_cast<bool>(checker), "AccuracyChecked: checker function must not be null");

    if (ref_compiled._ptr == nullptr) {
        return main_compiled;
    }

    auto cm = std::make_shared<CompiledModel>(model,
                                              plugin,
                                              std::move(main_compiled),
                                              std::move(ref_compiled),
                                              std::move(checker));
    return {cm, {}};
}

CompiledModel::CompiledModel(const std::shared_ptr<ov::Model>& model,
                             const std::shared_ptr<const ov::IPlugin>& plugin,
                             ov::SoPtr<ov::ICompiledModel> main_compiled,
                             ov::SoPtr<ov::ICompiledModel> ref_compiled,
                             Checker checker)
    : ov::ICompiledModel(model, plugin),
      m_main_compiled(std::move(main_compiled)),
      m_ref_compiled(std::move(ref_compiled)),
      m_checker(std::move(checker)) {}

ov::SoPtr<ov::ICompiledModel> CompiledModel::active_compiled_model_locked() const {
    return m_switched_to_reference ? m_ref_compiled : m_main_compiled;
}

bool CompiledModel::has_switched_to_reference() const {
    std::lock_guard<std::mutex> lock(m_mutex);
    return m_switched_to_reference;
}

void CompiledModel::export_model(std::ostream& stream) const {
    std::lock_guard<std::mutex> lock(m_mutex);
    active_compiled_model_locked()->export_model(stream);
}

std::shared_ptr<const ov::Model> CompiledModel::get_runtime_model() const {
    std::lock_guard<std::mutex> lock(m_mutex);
    return active_compiled_model_locked()->get_runtime_model();
}

void CompiledModel::set_property(const ov::AnyMap& properties) {
    std::lock_guard<std::mutex> lock(m_mutex);
    active_compiled_model_locked()->set_property(properties);
}

ov::Any CompiledModel::get_property(const std::string& name) const {
    std::lock_guard<std::mutex> lock(m_mutex);
    return active_compiled_model_locked()->get_property(name);
}

std::shared_ptr<ov::ISyncInferRequest> CompiledModel::create_sync_infer_request() const {
    auto self = std::static_pointer_cast<const CompiledModel>(shared_from_this());
    return std::make_shared<InferRequest>(std::move(self));
}

std::shared_ptr<ov::IAsyncInferRequest> CompiledModel::create_infer_request() const {
    return std::make_shared<ov::IAsyncInferRequest>(create_sync_infer_request(),
                                                    get_task_executor(),
                                                    get_callback_executor());
}

// ============================================================================
// InferRequest
// ============================================================================

InferRequest::InferRequest(std::shared_ptr<const CompiledModel> compiled_model)
    : ov::ISyncInferRequest(compiled_model),
      m_acc_compiled_model(std::move(compiled_model)),
      m_using_reference(m_acc_compiled_model->has_switched_to_reference()) {}

void InferRequest::ensure_main_request_locked() const {
    if (m_main_request) {
        return;
    }
    m_main_request = m_acc_compiled_model->m_main_compiled->create_infer_request();
    OPENVINO_ASSERT(m_main_request, "AccuracyChecked: failed to create main infer request");
}

void InferRequest::ensure_ref_request_locked() const {
    if (m_ref_request) {
        return;
    }
    m_ref_request = m_acc_compiled_model->m_ref_compiled->create_infer_request();
    OPENVINO_ASSERT(m_ref_request, "AccuracyChecked: failed to create reference infer request");
}

void InferRequest::infer() {
    std::lock_guard<std::mutex> lock(m_mutex);

    if (m_using_reference) {
        ensure_ref_request_locked();
        m_ref_request->infer();
        return;
    }

    ensure_main_request_locked();

    const auto& main_cm = m_acc_compiled_model->m_main_compiled;
    const auto& ref_cm = m_acc_compiled_model->m_ref_compiled;

    // Snapshot current input/output tensors from the main request.
    // These are the tensors the user has bound (or the request's defaults).
    // We use them to (a) feed the reference request and (b) rebind after a
    // permanent switch so that downstream requests keep reading from the
    // same memory.
    std::vector<ov::SoPtr<ov::ITensor>> main_input_tensors;
    std::vector<ov::SoPtr<ov::ITensor>> main_output_tensors;
    main_input_tensors.reserve(main_cm->inputs().size());
    main_output_tensors.reserve(main_cm->outputs().size());

    for (const auto& port : main_cm->inputs()) {
        main_input_tensors.push_back(m_main_request->get_tensor(port));
    }
    for (const auto& port : main_cm->outputs()) {
        main_output_tensors.push_back(m_main_request->get_tensor(port));
    }

    m_main_request->infer();

    // Accuracy check: feed reference request with the same inputs and run it.
    ensure_ref_request_locked();
    for (size_t i = 0; i < ref_cm->inputs().size(); i++) {
        m_ref_request->set_tensor(ref_cm->inputs()[i], main_input_tensors[i]);
    }
    m_ref_request->infer();

    // Compare outputs using the provided checker.
    bool accurate = true;
    for (size_t i = 0; i < main_cm->outputs().size(); i++) {
        const auto& ref_tensor = m_ref_request->get_tensor(ref_cm->outputs()[i]);
        if (!m_acc_compiled_model->m_checker(main_output_tensors[i], ref_tensor)) {
            accurate = false;
        }
    }

    if (!accurate) {
        // Copy reference outputs into the main output buffers so that any
        // downstream request already bound to those buffers sees the corrected
        // values immediately.
        for (size_t i = 0; i < main_cm->outputs().size(); i++) {
            const auto& ref_tensor = m_ref_request->get_tensor(ref_cm->outputs()[i]);
            ref_tensor->copy_to(main_output_tensors[i]._ptr);
        }

        // Rebind the reference request to use the same tensor objects that the
        // main request was using.  From this point on, the reference request
        // writes results directly into those buffers, keeping downstream
        // tensor bindings valid without requiring update_subrequest_links().
        for (size_t i = 0; i < ref_cm->inputs().size(); i++) {
            m_ref_request->set_tensor(ref_cm->inputs()[i], main_input_tensors[i]);
        }
        for (size_t i = 0; i < ref_cm->outputs().size(); i++) {
            m_ref_request->set_tensor(ref_cm->outputs()[i], main_output_tensors[i]);
        }

        m_using_reference = true;
        {
            std::lock_guard<std::mutex> model_lock(m_acc_compiled_model->m_mutex);
            m_acc_compiled_model->m_switched_to_reference = true;
        }
    }
}

ov::SoPtr<ov::ITensor> InferRequest::get_tensor(const ov::Output<const ov::Node>& port) const {
    std::lock_guard<std::mutex> lock(m_mutex);

    if (m_using_reference) {
        ensure_ref_request_locked();
        auto found = find_port(port);
        OPENVINO_ASSERT(found.found(), "AccuracyChecked: unknown port");
        const auto& ref_cm = m_acc_compiled_model->m_ref_compiled;
        return found.is_output() ? m_ref_request->get_tensor(ref_cm->outputs()[found.idx])
                                 : m_ref_request->get_tensor(ref_cm->inputs()[found.idx]);
    }

    ensure_main_request_locked();
    return m_main_request->get_tensor(port);
}

void InferRequest::set_tensor(const ov::Output<const ov::Node>& port, const ov::SoPtr<ov::ITensor>& tensor) {
    std::lock_guard<std::mutex> lock(m_mutex);

    if (m_using_reference) {
        ensure_ref_request_locked();
        auto found = find_port(port);
        OPENVINO_ASSERT(found.found(), "AccuracyChecked: unknown port");
        const auto& ref_cm = m_acc_compiled_model->m_ref_compiled;
        if (found.is_output()) {
            m_ref_request->set_tensor(ref_cm->outputs()[found.idx], tensor);
        } else {
            m_ref_request->set_tensor(ref_cm->inputs()[found.idx], tensor);
        }
        return;
    }

    ensure_main_request_locked();
    m_main_request->set_tensor(port, tensor);
}

void InferRequest::check_tensors() const {}

std::vector<ov::SoPtr<ov::IVariableState>> InferRequest::query_state() const {
    std::lock_guard<std::mutex> lock(m_mutex);
    if (m_using_reference) {
        ensure_ref_request_locked();
        return m_ref_request->query_state();
    }
    ensure_main_request_locked();
    return m_main_request->query_state();
}

std::vector<ov::ProfilingInfo> InferRequest::get_profiling_info() const {
    std::lock_guard<std::mutex> lock(m_mutex);
    if (m_using_reference) {
        ensure_ref_request_locked();
        return m_ref_request->get_profiling_info();
    }
    ensure_main_request_locked();
    return m_main_request->get_profiling_info();
}

}  // namespace ov::npuw::accuracy_checked
