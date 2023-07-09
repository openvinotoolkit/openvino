// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "sync_infer_request.hpp"

#include <algorithm>
#include <map>
#include <memory>
#include <string>
#include <utility>

#include "compiled_model.hpp"
#include "itt.hpp"
#include "openvino/core/except.hpp"
#include "openvino/op/util/variable_context.hpp"
#include "openvino/runtime/iasync_infer_request.hpp"
#include "openvino/runtime/ivariable_state.hpp"
#include "openvino/runtime/profiling_info.hpp"
#include "openvino/runtime/tensor.hpp"
#include "openvino/util/common_util.hpp"
#include "plugin.hpp"

// #include "template/remote_tensor.hpp"
// #include "variable_state.hpp"

using Time = std::chrono::high_resolution_clock;

ov::hetero::InferRequest::InferRequest(const std::shared_ptr<const ov::hetero::CompiledModel>& compiled_model)
    : ov::ISyncInferRequest(compiled_model) {
    int index = 0;
    for (auto&& subnetwork : compiled_model->m_networks) {
        InferRequest::SubRequestDesc desc;
        desc._network = subnetwork._network;

        std::string prof_task_name = compiled_model->m_name + "_Req" + std::to_string(index++);
        desc._profilingTask = {
            openvino::itt::handle("Hetero_" + prof_task_name + "_StartPipeline"),
            openvino::itt::handle("Hetero_" + prof_task_name + "_WaitPipline"),
        };
        desc._request = {desc._network->create_infer_request(), desc._network._so};
        m_infer_requests.push_back(desc);
    }

    for (size_t i = 0; i < compiled_model->inputs().size(); i++) {
        const auto& port = compiled_model->inputs()[i];
        const auto& submodel_idx = compiled_model->m_inputs_to_submodel_inputs[i].first;
        m_port_to_request_idx_map[port] = submodel_idx;
    }
    for (size_t i = 0; i < compiled_model->outputs().size(); i++) {
        const auto& port = compiled_model->outputs()[i];
        const auto& submodel_idx = compiled_model->m_outputs_to_submodel_outputs[i].first;
        m_port_to_request_idx_map[port] = submodel_idx;
    }

    for (const auto& kvp : compiled_model->m_submodels_input_to_prev_output) {
        const auto& submodel_idx_in = kvp.first.first;
        const auto& tensor_idx_in = kvp.first.second;
        const auto& submodel_idx_out = kvp.second.first;
        const auto& tensor_idx_out = kvp.second.second;

        const auto& output_port = m_infer_requests[submodel_idx_out]._network->outputs()[tensor_idx_out];
        const auto& output_tensor = m_infer_requests[submodel_idx_out]._request->get_tensor(output_port);
        const auto& input_port = m_infer_requests[submodel_idx_in]._network->inputs()[tensor_idx_in];
        m_infer_requests[submodel_idx_in]._request->set_tensor(input_port, output_tensor);
    }
}

ov::hetero::InferRequest::~InferRequest() = default;

ov::SoPtr<ov::IAsyncInferRequest> ov::hetero::InferRequest::get_request(const ov::Output<const ov::Node>& port) const {
    auto check_nodes = [](const ov::Node* node1, const ov::Node* node2) {
        return node1 == node2 ||
               (node1->get_friendly_name() == node2->get_friendly_name() &&
                node1->get_type_info() == node2->get_type_info() &&
                node1->outputs().size() == node2->outputs().size() && node1->inputs().size() == node2->inputs().size());
    };

    for (const auto& kvp : m_port_to_request_idx_map) {
        if (kvp.first.get_index() == port.get_index() && kvp.first.get_names() == port.get_names() &&
            check_nodes(kvp.first.get_node(), port.get_node())) {
            return m_infer_requests[kvp.second]._request;
        }
    }
    OPENVINO_THROW("Cannot find infer request for port ", port);
}

ov::Tensor ov::hetero::InferRequest::get_tensor(const ov::Output<const ov::Node>& port) const {
    return get_request(port)->get_tensor(port);
}

void ov::hetero::InferRequest::set_tensor(const ov::Output<const ov::Node>& port, const ov::Tensor& tensor) {
    get_request(port)->set_tensor(port, tensor);
}

std::vector<ov::Tensor> ov::hetero::InferRequest::get_tensors(const ov::Output<const ov::Node>& port) const {
    return get_request(port)->get_tensors(port);
}

void ov::hetero::InferRequest::set_tensors(const ov::Output<const ov::Node>& port,
                                        const std::vector<ov::Tensor>& tensors) {
    return get_request(port)->set_tensors(port, tensors);
}

void ov::hetero::InferRequest::check_tensors() const {
    // Ignore `check_tensor` of inputs and outputs of Hetero Compiled Model because
    // `m_tensors` are not allocated
    return;
}

std::vector<std::shared_ptr<ov::IVariableState>> ov::hetero::InferRequest::query_state() const {
    std::vector<std::shared_ptr<ov::IVariableState>> variable_states = {};
    for (auto&& desc : m_infer_requests) {
        auto& r = desc._request;
        assert(r);
        for (auto&& state : r->query_state()) {
            variable_states.emplace_back(state);
        }
    }
    return variable_states;
}

std::shared_ptr<const ov::hetero::CompiledModel> ov::hetero::InferRequest::get_hetero_model() const {
    auto& compiled_model = get_compiled_model();
    auto hetero_model = std::dynamic_pointer_cast<const ov::hetero::CompiledModel>(compiled_model);
    OPENVINO_ASSERT(hetero_model);
    return hetero_model;
}

void ov::hetero::InferRequest::infer() {
    start_pipeline();
    wait_pipeline();
}

void ov::hetero::InferRequest::start_pipeline() {
    auto start = Time::now();
    for (auto&& desc : m_infer_requests) {
        OV_ITT_SCOPED_TASK(itt::domains::Hetero, desc._profilingTask[StartPipeline])
        auto& request = desc._request;
        OPENVINO_ASSERT(request);
        request->infer();
    }
    m_durations[StartPipeline] = Time::now() - start;
}

void ov::hetero::InferRequest::wait_pipeline() {
    auto start = Time::now();
    for (auto&& desc : m_infer_requests) {
        OV_ITT_SCOPED_TASK(itt::domains::Hetero, desc._profilingTask[WaitPipeline])
        auto& request = desc._request;
        OPENVINO_ASSERT(request);
        request->wait();
    }
    m_durations[WaitPipeline] = Time::now() - start;
}

std::vector<ov::ProfilingInfo> ov::hetero::InferRequest::get_profiling_info() const {
    std::vector<ov::ProfilingInfo> info;
    const auto fill_profiling_info = [](const std::string& name,
                                        const std::chrono::duration<float, std::micro>& time) -> ov::ProfilingInfo {
        ov::ProfilingInfo p_info;
        p_info.status = ov::ProfilingInfo::Status::EXECUTED;
        p_info.node_name = name;
        p_info.cpu_time = p_info.real_time = std::chrono::duration_cast<std::chrono::milliseconds>(time);
        return p_info;
    };
    info.emplace_back(fill_profiling_info("execution time", m_durations[StartPipeline]));
    return info;
}

void ov::hetero::InferRequest::cancel() {
    for (auto&& desc : m_infer_requests) {
        desc._request->cancel();
    }
}