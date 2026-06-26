// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "batched.hpp"

#include <algorithm>
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

    // Snapshot the public inputs once -- read repeatedly below.
    std::vector<ov::SoPtr<ov::ITensor>> in_tensors;
    in_tensors.reserve(wrapper_inputs.size());
    for (const auto& port : wrapper_inputs) {
        in_tensors.push_back(get_tensor(port));
    }

    // Batch is the largest leading dim across inputs, so a shared [1, ...] input is broadcast
    // rather than mistaken for the batch. Stays 1 when nothing is batched.
    std::size_t batch = 1;
    bool any_batched = false;
    for (const auto& tensor : in_tensors) {
        if (!has_batch_dim(tensor)) {
            continue;
        }
        const std::size_t in_batch = tensor->get_shape()[0];
        batch = any_batched ? std::max(batch, in_batch) : in_batch;
        any_batched = true;
    }
    // A zero-sized batch has no rows and would publish unpopulated outputs; reject it.
    OPENVINO_ASSERT(batch > 0, "Batched element: batch size must be > 0, got an input with batch dimension 0.");

    // Every batched input must match `batch` or be a broadcast (dim 0 == 1); anything else
    // cannot be sliced per row.
    for (std::size_t i = 0; i < wrapper_inputs.size(); ++i) {
        if (!has_batch_dim(in_tensors[i])) {
            continue;
        }
        const std::size_t in_batch = in_tensors[i]->get_shape()[0];
        OPENVINO_ASSERT(in_batch == batch || in_batch == 1,
                        "Batched element: input '",
                        wrapper_inputs[i].get_any_name(),
                        "' has batch dimension ",
                        in_batch,
                        " which is neither the inferred batch size ",
                        batch,
                        " nor 1 (broadcast).");
    }

    // States are stable across rows: query once, reset per row so row i never sees row i-1.
    const auto inner_states = inner_query_state();
    const auto reset_inner_state = [&] {
        for (const auto& state : inner_states) {
            state->reset();
        }
    };

    // Slice per-row inputs out of [batch, ...]; pass broadcast/non-batched inputs through whole.
    const auto bind_row = [&](std::size_t row) {
        for (std::size_t i = 0; i < wrapper_inputs.size(); ++i) {
            const auto& full = in_tensors[i];
            const bool sliceable = batch > 1 && has_batch_dim(full) && full->get_shape()[0] == batch;
            inner_set_tensor(in_ports[i], sliceable ? row_slice(full, row) : full);
        }
    };

    // Per-row outputs stacked along axis 0. A single row has nothing to stack, so its output is
    // published as-is (no aggregation buffer or copy); a non-batched output collapses to the last row.
    std::vector<ov::SoPtr<ov::ITensor>> aggregated_outputs(wrapper_outputs.size());

    for (std::size_t row = 0; row < batch; ++row) {
        reset_inner_state();
        bind_row(row);
        inner_infer();

        for (std::size_t i = 0; i < wrapper_outputs.size(); ++i) {
            const auto inner_out = inner_get_tensor(out_ports[i]);
            if (batch == 1) {
                aggregated_outputs[i] = inner_out;
                continue;
            }
            if (!aggregated_outputs[i]) {
                ov::Shape out_shape = inner_out->get_shape();
                if (!out_shape.empty()) {
                    out_shape[0] = batch;
                }
                aggregated_outputs[i] = ov::get_tensor_impl(ov::Tensor(inner_out->get_element_type(), out_shape));
            }
            if (has_batch_dim(inner_out)) {
                inner_out->copy_to(row_slice(aggregated_outputs[i], row)._ptr);
            } else {
                inner_out->copy_to(aggregated_outputs[i]._ptr);
            }
        }
    }

    for (std::size_t i = 0; i < wrapper_outputs.size(); ++i) {
        set_tensor(wrapper_outputs[i], aggregated_outputs[i]);
    }
}

void ov::npuw::batched::InferRequest::check_tensors() const {
    // No-op: batched tensors are unrolled and validated per row by the inner request.
}

std::vector<ov::SoPtr<ov::IVariableState>> ov::npuw::batched::InferRequest::query_state() const {
    // The batched element resets inner state between rows and exposes no
    // cross-call state of its own, so it presents an empty state list.
    return {};
}

std::vector<ov::ProfilingInfo> ov::npuw::batched::InferRequest::get_profiling_info() const {
    return {};
}
