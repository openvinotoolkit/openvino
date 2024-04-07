// Copyright (C) 2018-2024 Intel Corporation
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
#include "openvino/runtime/make_tensor.hpp"
#include "plugin.hpp"

ov::hetero::InferRequest::InferRequest(const std::shared_ptr<const ov::hetero::CompiledModel>& compiled_model)
    : ov::ISyncInferRequest(compiled_model) {
    for (auto&& comp_model_desc : compiled_model->m_compiled_submodels) {
        auto& comp_model = comp_model_desc.compiled_model;
        m_subrequests.push_back({comp_model->create_infer_request(), comp_model._so});
    }

    for (size_t i = 0; i < compiled_model->inputs().size(); i++) {
        const auto& port = compiled_model->inputs()[i];
        const auto& submodel_idx = compiled_model->m_mapping_info._inputs_to_submodels_inputs[i].first;
        m_port_to_subrequest_idx[port] = submodel_idx;
    }
    for (size_t i = 0; i < compiled_model->outputs().size(); i++) {
        const auto& port = compiled_model->outputs()[i];
        const auto& submodel_idx = compiled_model->m_mapping_info._outputs_to_submodels_outputs[i].first;
        m_port_to_subrequest_idx[port] = submodel_idx;
    }

    std::map<ov::Output<const ov::Node>, ov::SoPtr<ov::ITensor>> temp_tensor_map;
    for (const auto& kvp : compiled_model->m_mapping_info._submodels_input_to_prev_output) {
        const auto& submodel_idx_in = kvp.first.first;
        const auto& port_idx_in = kvp.first.second;
        const auto& submodel_idx_out = kvp.second.first;
        const auto& port_idx_out = kvp.second.second;

        const auto& output_port = m_subrequests[submodel_idx_out]->get_compiled_model()->outputs()[port_idx_out];
        const auto& output_tensor = m_subrequests[submodel_idx_out]->get_tensor(output_port);
        if (temp_tensor_map.find(output_port) == temp_tensor_map.end()) {
            temp_tensor_map[output_port] = {
                ov::make_tensor(output_tensor->get_element_type(), output_tensor->get_shape()),
                nullptr};
        }
        m_subrequests[submodel_idx_out]->set_tensor(output_port, temp_tensor_map[output_port]);
        const auto& input_port = m_subrequests[submodel_idx_in]->get_compiled_model()->inputs()[port_idx_in];
        m_subrequests[submodel_idx_in]->set_tensor(input_port, temp_tensor_map[output_port]);
    }
}

ov::hetero::InferRequest::~InferRequest() = default;

ov::SoPtr<ov::IAsyncInferRequest> ov::hetero::InferRequest::get_request(const ov::Output<const ov::Node>& port) const {
    auto found_port = find_port(port);
    ov::Output<const ov::Node> internal_port;
    OPENVINO_ASSERT(found_port.found(), "Cannot find infer request for port ", port);
    if (found_port.is_input()) {
        internal_port = get_inputs().at(found_port.idx);
    } else {
        internal_port = get_outputs().at(found_port.idx);
    }
    return m_subrequests[m_port_to_subrequest_idx.at(internal_port)];
}

ov::SoPtr<ov::ITensor> ov::hetero::InferRequest::get_tensor(const ov::Output<const ov::Node>& port) const {
    const auto infer_request = get_request(port);
    auto tensor = infer_request->get_tensor(port);
    if (!tensor._so) {
        tensor._so = infer_request._so;
    }
    return tensor;
}

void ov::hetero::InferRequest::set_tensor(const ov::Output<const ov::Node>& port,
                                          const ov::SoPtr<ov::ITensor>& tensor) {
    get_request(port)->set_tensor(port, tensor);
}

std::vector<ov::SoPtr<ov::ITensor>> ov::hetero::InferRequest::get_tensors(
    const ov::Output<const ov::Node>& port) const {
    const auto infer_request = get_request(port);
    auto tensors = infer_request->get_tensors(port);
    for (auto& tensor : tensors) {
        if (!tensor._so) {
            tensor._so = infer_request._so;
        }
    }
    return tensors;
}

void ov::hetero::InferRequest::set_tensors(const ov::Output<const ov::Node>& port,
                                           const std::vector<ov::SoPtr<ov::ITensor>>& tensors) {
    return get_request(port)->set_tensors(port, tensors);
}

void ov::hetero::InferRequest::check_tensors() const {
    // Ignore `check_tensor` of inputs and outputs of Hetero Compiled Model because
    // `m_tensors` are not allocated
    return;
}

std::vector<ov::SoPtr<ov::IVariableState>> ov::hetero::InferRequest::query_state() const {
    std::vector<ov::SoPtr<ov::IVariableState>> variable_states = {};
    for (const auto& request : m_subrequests) {
        OPENVINO_ASSERT(request);
        for (auto&& state : request->query_state()) {
            if (!state._so)
                state._so = request._so;
            variable_states.emplace_back(state);
        }
    }
    return variable_states;
}

void ov::hetero::InferRequest::infer() {
    for (auto&& request : m_subrequests) {
        OPENVINO_ASSERT(request);
        request->infer();
    }
}

std::vector<ov::ProfilingInfo> ov::hetero::InferRequest::get_profiling_info() const {
    std::vector<ov::ProfilingInfo> info;
    for (size_t i = 0; i < m_subrequests.size(); ++i) {
        auto&& subreq_info = m_subrequests[i]->get_profiling_info();
        for (auto&& rec : subreq_info)
            rec.node_name = std::string("subgraph") + std::to_string(i) + ": " + rec.node_name;
        info.insert(info.end(), subreq_info.begin(), subreq_info.end());
    }
    return info;
}
