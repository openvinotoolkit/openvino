// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "sync_infer_request.hpp"

#include <algorithm>
#include <map>
#include <memory>
#include <string>
#include <utility>

#include "itt.hpp"
#include "plugin.hpp"
#include "compiled_model.hpp"

#include "openvino/core/except.hpp"
#include "openvino/op/util/variable_context.hpp"
#include "openvino/runtime/ivariable_state.hpp"
#include "openvino/runtime/profiling_info.hpp"
#include "openvino/runtime/tensor.hpp"
#include "openvino/runtime/iasync_infer_request.hpp"
#include "openvino/util/common_util.hpp"

// #include "template/remote_tensor.hpp"
// #include "variable_state.hpp"

using Time = std::chrono::high_resolution_clock;

namespace {

void allocate_tensor_impl(ov::Tensor& tensor, const ov::element::Type& element_type, const ov::Shape& shape) {
    if (!tensor || tensor.get_element_type() != element_type) {
        tensor = ov::Tensor(element_type, shape);
    } else {
        tensor.set_shape(shape);
    }
}

ov::SoPtr<ov::IAsyncInferRequest> find_request_for_port(const ov::Output<const ov::Node>& port, std::map<ov::Output<const ov::Node>, ov::SoPtr<ov::IAsyncInferRequest>> port_to_request_map) {
    auto check_nodes = [](const ov::Node* node1, const ov::Node* node2) {
        return node1 == node2 ||
               (node1->get_friendly_name() == node2->get_friendly_name() &&
                node1->get_type_info() == node2->get_type_info() &&
                node1->outputs().size() == node2->outputs().size() && node1->inputs().size() == node2->inputs().size());
    };
    
    for (const auto& kvp : port_to_request_map) {
        // TODO: Fix port comparison
        // if (kvp.first == port) {
        if (kvp.first.get_index() == port.get_index() && kvp.first.get_names() == port.get_names() &&
            check_nodes(kvp.first.get_node(), port.get_node())) {
            return kvp.second;
        }
    }
    return ov::SoPtr<ov::IAsyncInferRequest>(nullptr, nullptr);
}

std::pair<bool, ov::Output<const ov::Node>> find_port_from_map(const ov::Output<const ov::Node>& port, std::map<ov::Output<const ov::Node>, ov::Output<const ov::Node>> input_to_output_map) {
    auto check_nodes = [](const ov::Node* node1, const ov::Node* node2) {
        return node1 == node2 ||
               (node1->get_friendly_name() == node2->get_friendly_name() &&
                node1->get_type_info() == node2->get_type_info() &&
                node1->outputs().size() == node2->outputs().size() && node1->inputs().size() == node2->inputs().size());
    };
    
    for (const auto& kvp : input_to_output_map) {
        // TODO: Fix port comparison
        // if (kvp.first == port) {
        if (kvp.first.get_index() == port.get_index() && kvp.first.get_names() == port.get_names() &&
            check_nodes(kvp.first.get_node(), port.get_node())) {
            return {true, kvp.second};
        }
    }
    return {false , {}};
}

}  // namespace

ov::hetero::InferRequest::InferRequest(const std::shared_ptr<const ov::hetero::CompiledModel>& compiled_model)
    : ov::ISyncInferRequest(compiled_model) {
    int index = 0;
    for (auto&& subnetwork : compiled_model->m_networks) {
        InferRequest::SubRequestDesc desc;
        desc._network = subnetwork._network;
        
        std::string prof_task_name = get_hetero_model()->m_model->get_friendly_name() + "_Req" + std::to_string(index);
        desc._profilingTask = {
            openvino::itt::handle("Hetero_" + prof_task_name + "_StartPipeline"),
            openvino::itt::handle("Hetero_" + prof_task_name + "_WaitPipline"),
        };
        desc._request = {desc._network->create_infer_request(), desc._network._so};
        
        auto requestBlob([&](const ov::Output<const ov::Node>& port, ov::SoPtr<ov::IAsyncInferRequest>& r, bool output) {
            auto subgraphInputToOutputBlobNames = get_hetero_model()->_blobNameMap;
            auto intermediateBlobName = port;
            
            // >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
            // TODO VURUSOVS CONTINUE FROM HERE !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            // !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            // >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>


            // auto itName = subgraphInputToOutputBlobNames.find(port);    // TODO vurusovs FIND (AND MAYBE INSERT) CAN NOT WORK FOR PORTS
            // if (itName != subgraphInputToOutputBlobNames.end()) {
            //     intermediateBlobName = itName->second;
            // }
            auto res = find_port_from_map(port, subgraphInputToOutputBlobNames);
            if (std::get<0>(res)) {
                intermediateBlobName = std::get<1>(res);
            }
            if (output) {
                // TODO: Fix port comparison
                // if (ov::util::contains(get_outputs(), port)) {   // ov::util::contains doesn't work because == not work
                if (find_port(port).found()) {
                    m_port_to_request_map.emplace(port, r);
                } else {
                    m_port_to_tensor_map.emplace(intermediateBlobName, r->get_tensor(port));
                }
            } else {
                // TODO: Fix port comparison
                // if (ov::util::contains(get_inputs(), port)) {
                if (find_port(port).found()) {
                    m_port_to_request_map.emplace(port, r);
                } else {
                    r->set_tensor(port, m_port_to_tensor_map.at(intermediateBlobName));
                }
            }
        });
        
        
        for (auto&& output : desc._network->outputs()) {
            requestBlob(output,  desc._request, true);
        }

        for (auto&& input : desc._network->inputs()) {
            requestBlob(input,  desc._request, false);
        }
        
        m_infer_requests.push_back(desc);
    }
}

ov::hetero::InferRequest::~InferRequest() = default;


ov::Tensor ov::hetero::InferRequest::get_tensor(const ov::Output<const ov::Node>& port) const {
    // auto request_it = m_port_to_request_map.find(port); // TODO: Fix port comparison
    // OPENVINO_ASSERT(request_it != m_port_to_request_map.end(), "TODO vurusovs PROVIDE TEXT");
    // return request_it->second->get_tensor(port);

    auto request = ::find_request_for_port(port, m_port_to_request_map);
    OPENVINO_ASSERT(request, "TODO vurusovs PROVIDE TEXT");
    return request->get_tensor(port);
}


void ov::hetero::InferRequest::set_tensor(const ov::Output<const ov::Node>& port, const ov::Tensor& tensor) {
    // auto request_it = m_port_to_request_map.find(port); // TODO: Fix port comparison
    // OPENVINO_ASSERT(request_it != m_port_to_request_map.end(), "TODO vurusovs PROVIDE TEXT");
    // request_it->second->set_tensor(port, tensor);

    auto request = ::find_request_for_port(port, m_port_to_request_map);
    OPENVINO_ASSERT(request, "TODO vurusovs PROVIDE TEXT");
    request->set_tensor(port, tensor);
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

std::shared_ptr<const ov::hetero::CompiledModel> ov::hetero::InferRequest::get_hetero_model()
    const {
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
