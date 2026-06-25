// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "batched.hpp"

#include <utility>

#include "openvino/core/except.hpp"
#include "openvino/runtime/make_tensor.hpp"
#include "openvino/runtime/tensor.hpp"

namespace {

// Zero-copy view of one batch row (axis 0) of a tensor: [N, ...] -> [1, ...].
ov::SoPtr<ov::ITensor> row_slice(const ov::SoPtr<ov::ITensor>& tensor, std::size_t row) {
    ov::Shape start_shape(tensor->get_shape().size(), 0u);
    start_shape[0] = row;
    ov::Shape end_shape = tensor->get_shape();
    end_shape[0] = row + 1;
    return ov::get_tensor_impl(ov::Tensor(ov::make_tensor(tensor), start_shape, end_shape));
}

bool has_batch_dim(const ov::SoPtr<ov::ITensor>& tensor) {
    return !tensor->get_shape().empty();
}

}  // namespace

ov::SoPtr<ov::ICompiledModel> ov::npuw::batched::CompiledModel::create(
    const std::shared_ptr<ov::Model>& model,
    const std::shared_ptr<const ov::IPlugin>& plugin,
    ov::SoPtr<ov::ICompiledModel> inner_compiled,
    bool enabled) {
    OPENVINO_ASSERT(inner_compiled._ptr != nullptr, "Batched compiled model requires an inner compiled model");

    // No-op wrapper: hand back the inner model unchanged for the zero-overhead path.
    if (!enabled) {
        return inner_compiled;
    }

    auto compiled_model = std::make_shared<CompiledModel>(model, plugin, std::move(inner_compiled));
    return {compiled_model, {}};
}

ov::npuw::batched::CompiledModel::CompiledModel(const std::shared_ptr<ov::Model>& model,
                                                const std::shared_ptr<const ov::IPlugin>& plugin,
                                                ov::SoPtr<ov::ICompiledModel> inner_compiled)
    : ov::ICompiledModel(model, plugin),
      m_inner(std::move(inner_compiled)) {
    OPENVINO_ASSERT(m_inner._ptr != nullptr, "Batched compiled model requires an inner compiled model");
}

void ov::npuw::batched::CompiledModel::export_model(std::ostream& model) const {
    m_inner->export_model(model);
}

std::shared_ptr<const ov::Model> ov::npuw::batched::CompiledModel::get_runtime_model() const {
    return m_inner->get_runtime_model();
}

void ov::npuw::batched::CompiledModel::set_property(const ov::AnyMap& properties) {
    m_inner->set_property(properties);
}

ov::Any ov::npuw::batched::CompiledModel::get_property(const std::string& name) const {
    return m_inner->get_property(name);
}

std::shared_ptr<ov::ISyncInferRequest> ov::npuw::batched::CompiledModel::create_sync_infer_request() const {
    auto self = std::static_pointer_cast<const ov::ICompiledModel>(shared_from_this());
    auto inner_request = m_inner->create_infer_request();
    OPENVINO_ASSERT(inner_request != nullptr, "Batched element: inner compiled model returned a null request");
    return std::make_shared<InferRequest>(self, std::move(inner_request));
}

std::shared_ptr<ov::IAsyncInferRequest> ov::npuw::batched::CompiledModel::create_infer_request() const {
    return std::make_shared<ov::IAsyncInferRequest>(create_sync_infer_request(),
                                                    get_task_executor(),
                                                    get_callback_executor());
}

ov::npuw::batched::InferRequest::InferRequest(const std::shared_ptr<const ov::ICompiledModel>& compiled_model,
                                              std::shared_ptr<ov::IAsyncInferRequest> inner_request)
    : ov::ISyncInferRequest(compiled_model),
      m_inner_async(std::move(inner_request)) {
    OPENVINO_ASSERT(m_inner_async != nullptr, "Batched element requires a non-null inner request");
    init_public_tensors();
}

ov::npuw::batched::InferRequest::InferRequest(const std::shared_ptr<const ov::ICompiledModel>& compiled_model,
                                              std::shared_ptr<ov::ISyncInferRequest> inner_request)
    : ov::ISyncInferRequest(compiled_model),
      m_inner_sync(std::move(inner_request)) {
    OPENVINO_ASSERT(m_inner_sync != nullptr, "Batched element requires a non-null inner request");
    init_public_tensors();
}

void ov::npuw::batched::InferRequest::init_public_tensors() {
    // Allocate a tensor for every public port so get_tensor() never returns an
    // uninitialized handle (callers such as the Python infer(dict) dispatcher fetch
    // the tensor before populating it). Dynamic dims are sized to 0 and resized later;
    // the real [N, ...] tensors are bound by the caller (inputs) or by infer() (outputs).
    const auto init_port = [this](const ov::Output<const ov::Node>& port) {
        if (ov::ISyncInferRequest::get_tensor(port)) {
            return;
        }
        const auto& pshape = port.get_partial_shape();
        ov::Shape shape;
        if (pshape.is_dynamic()) {
            for (const auto& dim : pshape) {
                shape.push_back(dim.is_static() ? dim.get_length() : 0);
            }
        } else {
            shape = pshape.to_shape();
        }
        set_tensor(port, ov::get_tensor_impl(ov::Tensor(port.get_element_type(), shape)));
    };
    for (const auto& port : get_inputs()) {
        init_port(port);
    }
    for (const auto& port : get_outputs()) {
        init_port(port);
    }
}

const std::vector<ov::Output<const ov::Node>>& ov::npuw::batched::InferRequest::inner_inputs() const {
    return m_inner_sync ? m_inner_sync->get_inputs() : m_inner_async->get_inputs();
}

const std::vector<ov::Output<const ov::Node>>& ov::npuw::batched::InferRequest::inner_outputs() const {
    return m_inner_sync ? m_inner_sync->get_outputs() : m_inner_async->get_outputs();
}

void ov::npuw::batched::InferRequest::inner_set_tensor(const ov::Output<const ov::Node>& port,
                                                       const ov::SoPtr<ov::ITensor>& tensor) {
    if (m_inner_sync) {
        m_inner_sync->set_tensor(port, tensor);
    } else {
        m_inner_async->set_tensor(port, tensor);
    }
}

ov::SoPtr<ov::ITensor> ov::npuw::batched::InferRequest::inner_get_tensor(const ov::Output<const ov::Node>& port) const {
    return m_inner_sync ? m_inner_sync->get_tensor(port) : m_inner_async->get_tensor(port);
}

void ov::npuw::batched::InferRequest::inner_infer() {
    if (m_inner_sync) {
        m_inner_sync->infer();
    } else {
        m_inner_async->infer();
    }
}

std::vector<ov::SoPtr<ov::IVariableState>> ov::npuw::batched::InferRequest::inner_query_state() const {
    return m_inner_sync ? m_inner_sync->query_state() : m_inner_async->query_state();
}

void ov::npuw::batched::InferRequest::infer() {
    std::lock_guard<std::mutex> lock(m_mutex);

    const auto& wrapper_inputs = get_inputs();
    const auto& wrapper_outputs = get_outputs();
    const auto& in_ports = inner_inputs();
    const auto& out_ports = inner_outputs();

    OPENVINO_ASSERT(wrapper_inputs.size() == in_ports.size() && wrapper_outputs.size() == out_ports.size(),
                    "Batched element: inner request I/O does not match the wrapped model");

    // Batch size is taken from the first input that carries a batch dimension.
    std::size_t batch = 1;
    for (const auto& port : wrapper_inputs) {
        const auto tensor = get_tensor(port);
        if (has_batch_dim(tensor)) {
            batch = tensor->get_shape()[0];
            break;
        }
    }
    // A zero-sized batch has no rows to score and would leave the outputs unpopulated
    // (publishing null tensors). Reject it explicitly rather than silently no-op.
    OPENVINO_ASSERT(batch > 0, "Batched element: batch size must be > 0, got an input with batch dimension 0.");

    // Validate every batched input up front: each row-carrying input must have a batch
    // dimension equal to `batch`, or exactly 1 (a shared/broadcast input passed to every row).
    // Anything else (e.g. a [M, ...] input with M != batch and M != 1) cannot be sliced per row
    // and would otherwise be fed whole into the batch-1 inner request -> wrong results.
    for (std::size_t i = 0; i < wrapper_inputs.size(); ++i) {
        const auto full = get_tensor(wrapper_inputs[i]);
        if (!has_batch_dim(full)) {
            continue;
        }
        const std::size_t in_batch = full->get_shape()[0];
        OPENVINO_ASSERT(in_batch == batch || in_batch == 1,
                        "Batched element: input '",
                        wrapper_inputs[i].get_any_name(),
                        "' has batch dimension ",
                        in_batch,
                        " which is neither the inferred batch size ",
                        batch,
                        " nor 1 (broadcast).");
    }

    // Aggregated [batch, ...] outputs, allocated lazily once the per-row output shape is known.
    std::vector<ov::SoPtr<ov::ITensor>> aggregated_outputs(wrapper_outputs.size());

    // The inner request's variable states are stable across rows, so query them once and reset
    // them per row rather than re-querying (which may allocate) on every iteration.
    const auto inner_states = inner_query_state();

    for (std::size_t row = 0; row < batch; ++row) {
        // Rows are independent prompts: clear the inner request's variable state
        // (KV-cache) so row i never sees row i-1.  Harmless for stateless inners.
        for (const auto& state : inner_states) {
            state->reset();
        }

        // Bind the row's slice of every batched input; pass shared ([1, ...] or non-batched)
        // inputs through unchanged.
        for (std::size_t i = 0; i < wrapper_inputs.size(); ++i) {
            const auto full = get_tensor(wrapper_inputs[i]);
            const bool sliceable = batch > 1 && has_batch_dim(full) && full->get_shape()[0] == batch;
            inner_set_tensor(in_ports[i], sliceable ? row_slice(full, row) : full);
        }

        inner_infer();

        // Stack each per-row output into row i of the aggregated [batch, ...] tensor.
        for (std::size_t i = 0; i < wrapper_outputs.size(); ++i) {
            const auto inner_out = inner_get_tensor(out_ports[i]);
            if (!aggregated_outputs[i]) {
                ov::Shape out_shape = inner_out->get_shape();
                if (!out_shape.empty()) {
                    out_shape[0] = batch;
                }
                aggregated_outputs[i] = ov::get_tensor_impl(ov::Tensor(inner_out->get_element_type(), out_shape));
            }
            if (batch > 1 && has_batch_dim(inner_out)) {
                const auto slot = row_slice(aggregated_outputs[i], row);
                inner_out->copy_to(slot._ptr);
            } else {
                inner_out->copy_to(aggregated_outputs[i]._ptr);
            }
        }
    }

    // Publish the aggregated outputs as this request's public output tensors.
    for (std::size_t i = 0; i < wrapper_outputs.size(); ++i) {
        set_tensor(wrapper_outputs[i], aggregated_outputs[i]);
    }
}

void ov::npuw::batched::InferRequest::check_tensors() const {}

std::vector<ov::SoPtr<ov::IVariableState>> ov::npuw::batched::InferRequest::query_state() const {
    // The batched element resets inner state between rows and exposes no
    // cross-call state of its own, so it presents an empty state list.
    return {};
}

std::vector<ov::ProfilingInfo> ov::npuw::batched::InferRequest::get_profiling_info() const {
    return {};
}
