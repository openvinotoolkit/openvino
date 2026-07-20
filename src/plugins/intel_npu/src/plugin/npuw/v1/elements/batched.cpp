// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "batched.hpp"

#include <algorithm>
#include <utility>

#include "../../logging.hpp"
#include "../../util.hpp"
#include "intel_npu/npuw_private_properties.hpp"
#include "openvino/core/except.hpp"
#include "openvino/runtime/make_tensor.hpp"
#include "openvino/runtime/tensor.hpp"

bool ov::npuw::batched::requested(const std::shared_ptr<ov::npuw::ICompiledModel>& model) {
    OPENVINO_ASSERT(model != nullptr, "Batched element: null compiled model");
    const auto is_enabled = [&model](const std::string& key) {
        return model->get_property(key).as<bool>();
    };
    return is_enabled(ov::intel_npu::npuw::text_rerank::enabled.name()) ||
           is_enabled(ov::intel_npu::npuw::text_embed::enabled.name());
}

ov::npuw::batched::CompiledModel::CompiledModel(const std::shared_ptr<ov::npuw::ICompiledModel>& inner,
                                                const std::shared_ptr<const ov::IPlugin>& plugin)
    : ov::npuw::ICompiledModel(nullptr, plugin),  // I/O comes from the inner via inputs()/outputs()
      m_inner(inner) {
    OPENVINO_ASSERT(m_inner != nullptr, "Batched compiled model requires an inner compiled model");
}

const std::vector<ov::Output<const ov::Node>>& ov::npuw::batched::CompiledModel::inputs() const {
    return m_inner->inputs();
}

const std::vector<ov::Output<const ov::Node>>& ov::npuw::batched::CompiledModel::outputs() const {
    return m_inner->outputs();
}

void ov::npuw::batched::CompiledModel::export_model(std::ostream& model) const {
    // The element is a runtime-only decorator: the blob is the inner's blob, and the
    // entry points re-apply the wrapper on import based on the properties.
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

void ov::npuw::batched::CompiledModel::release_memory() {
    m_inner->release_memory();
}

std::shared_ptr<ov::ISyncInferRequest> ov::npuw::batched::CompiledModel::create_sync_infer_request() const {
    auto self = std::static_pointer_cast<const ov::ICompiledModel>(shared_from_this());
    auto inner_request = m_inner->create_infer_request();
    OPENVINO_ASSERT(inner_request != nullptr, "Batched element: inner compiled model returned a null request");
    return std::make_shared<InferRequest>(self, std::move(inner_request));
}

ov::npuw::batched::InferRequest::InferRequest(const std::shared_ptr<const ov::ICompiledModel>& compiled_model,
                                              std::shared_ptr<ov::IAsyncInferRequest> inner_request)
    : ov::ISyncInferRequest(compiled_model),
      m_inner(std::move(inner_request)) {
    OPENVINO_ASSERT(m_inner != nullptr, "Batched element requires a non-null inner request");

    m_profile.report_on_die = ov::npuw::profiling_enabled();
    m_profile.area = "batched/execution";

    // Surface the inner request's own tensors as the public defaults (the ports are
    // the same objects, see CompiledModel::inputs()). Nothing is allocated here: a
    // batch-1 caller works directly on the inner's tensors, and a batched caller
    // replaces them with its [N, ...] tensors via set_tensor().
    for (const auto& port : get_inputs()) {
        if (auto tensor = m_inner->get_tensor(port)) {
            set_tensor(port, tensor);
        }
    }
}

ov::npuw::batched::InferRequest::BatchedInputs ov::npuw::batched::InferRequest::extract_batch() const {
    const auto& in_ports = get_inputs();
    OPENVINO_ASSERT(!in_ports.empty(), "Batched element: the wrapped model has no inputs");

    BatchedInputs inputs;
    inputs.tensors.reserve(in_ports.size());
    for (const auto& port : in_ports) {
        auto tensor = get_tensor(port);
        OPENVINO_ASSERT(tensor, "Batched element: no tensor is set for input '", port.get_any_name(), "'");
        const auto& shape = tensor->get_shape();
        OPENVINO_ASSERT(!shape.empty(),
                        "Batched element: input '",
                        port.get_any_name(),
                        "' has no leading (batch) dimension");
        OPENVINO_ASSERT(shape[0] > 0,
                        "Batched element: input '",
                        port.get_any_name(),
                        "' has a zero-sized batch dimension - batch size must be > 0");
        inputs.batch = std::max(inputs.batch, shape[0]);
        inputs.tensors.push_back(std::move(tensor));
    }
    for (std::size_t i = 0; i < in_ports.size(); ++i) {
        const std::size_t in_batch = inputs.tensors[i]->get_shape()[0];
        OPENVINO_ASSERT(in_batch == inputs.batch || in_batch == 1,
                        "Batched element: input '",
                        in_ports[i].get_any_name(),
                        "' has batch dimension ",
                        in_batch,
                        " which is neither the inferred batch size ",
                        inputs.batch,
                        " nor 1 (shared).");
    }
    return inputs;
}

void ov::npuw::batched::InferRequest::infer() {
    std::lock_guard<std::mutex> lock(m_mutex);

    const auto& in_ports = get_inputs();
    const auto& out_ports = get_outputs();

    BatchedInputs inputs;
    m_profile["1.extract_batch"].record([&]() {
        inputs = extract_batch();
    });
    const std::size_t batch = inputs.batch;

    // Unroll row by row: reset the inner variable state so each row is scored as an
    // independent prompt, bind the row's [1, ...] view of every batched input, run
    // the batch-1 inner request, and write the row's outputs into row `row` of the
    // [N, ...] public output tensors.
    const auto inner_states = m_inner->query_state();
    for (std::size_t row = 0; row < batch; ++row) {
        m_profile["2.bind_row"].record([&]() {
            for (const auto& state : inner_states) {
                state->reset();
            }
            for (std::size_t i = 0; i < in_ports.size(); ++i) {
                const auto& full = inputs.tensors[i];
                m_inner->set_tensor(in_ports[i],
                                    full->get_shape()[0] == 1 ? full : ov::npuw::util::view(full, 0, row, 1));
            }
        });
        m_profile["3.inner_infer"].record([&]() {
            m_inner->infer();
        });

        if (row == 0) {
            // The wrapped model's ports are dynamic - the output shapes are only
            // known once the first row has been scored.
            ensure_batched_outputs(batch);
        }
        m_profile["4.copy_row_out"].record([&]() {
            for (const auto& port : out_ports) {
                m_inner->get_tensor(port)->copy_to(ov::npuw::util::view(get_tensor(port), 0, row, 1)._ptr);
            }
        });
    }
}

void ov::npuw::batched::InferRequest::ensure_batched_outputs(std::size_t batch) {
    for (const auto& port : get_outputs()) {
        const auto inner_out = m_inner->get_tensor(port);
        OPENVINO_ASSERT(inner_out && !inner_out->get_shape().empty() && inner_out->get_shape()[0] == 1,
                        "Batched element: output '",
                        port.get_any_name(),
                        "' of the inner request is not a [1, ...] tensor");
        ov::Shape shape = inner_out->get_shape();
        shape[0] = batch;
        const auto current = get_tensor(port);
        if (!current || current->get_element_type() != inner_out->get_element_type() || current->get_shape() != shape) {
            set_tensor(port, ov::get_tensor_impl(ov::Tensor(inner_out->get_element_type(), shape)));
        }
    }
}

void ov::npuw::batched::InferRequest::check_tensors() const {
    // No-op: the public outputs are late-bound (allocated on infer once the batch is
    // known), and the batched inputs are validated and then unrolled by infer() -- the
    // per-row [1, ...] tensors are checked by the inner request.
}

std::vector<ov::SoPtr<ov::IVariableState>> ov::npuw::batched::InferRequest::query_state() const {
    // The batched element resets inner state between rows and exposes no
    // cross-call state of its own, so it presents an empty state list.
    return {};
}

std::vector<ov::ProfilingInfo> ov::npuw::batched::InferRequest::get_profiling_info() const {
    return m_inner->get_profiling_info();
}
