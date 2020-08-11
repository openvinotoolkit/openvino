// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <list>
#include <set>
#include <unordered_set>

#include "ie_metric_helpers.hpp"
#include <api/cldnn.hpp>
#include <api/data.hpp>
#include <chrono>
#include <cmath>
#include <algorithm>
#include "cldnn_graph.h"

#include <description_buffer.hpp>
#include <cldnn/cldnn_config.hpp>
#include <legacy/graph_tools.hpp>
#include <legacy/ie_layers_internal.hpp>
#include <legacy/net_pass.h>
#include "cldnn_infer_request.h"
#include <threading/ie_executor_manager.hpp>
#include "cldnn_async_infer_request.h"
#include <fstream>
#include <utility>
#include <sys/types.h>

#include "cldnn_executable_network.h"
#include "threading/ie_cpu_streams_executor.hpp"


using namespace InferenceEngine;
using namespace InferenceEngine::details;

namespace CLDNNPlugin {

CLDNNExecNetwork::CLDNNExecNetwork(InferenceEngine::ICNNNetwork &network, RemoteContext::Ptr context, Config config) :
    InferenceEngine::ExecutableNetworkThreadSafeDefault{[&]()->InferenceEngine::ITaskExecutor::Ptr {
        if (config.throughput_streams > 1) {
            return std::make_shared<InferenceEngine::CPUStreamsExecutor>(
                IStreamsExecutor::Config{"CLDNNPlugin executor", config.throughput_streams});
        } else if (config.exclusiveAsyncRequests) {
            return ExecutorManager::getInstance()->getExecutor("GPU");
        } else {
            return std::make_shared<InferenceEngine::CPUStreamsExecutor>(
                IStreamsExecutor::Config{"CLDNNPlugin executor", 1});
        }
    }()},
    m_config(config),
    m_taskExecutor{_taskExecutor} {
    auto casted_context = std::dynamic_pointer_cast<gpu::ClContext>(context);

    if (nullptr == casted_context) {
        THROW_IE_EXCEPTION << "Invalid remote context";
    }

    m_context = casted_context;

    auto graph_base = std::make_shared<CLDNNGraph>(network, m_context, m_config, 0);
    for (uint16_t n = 0; n < m_config.throughput_streams; n++) {
        auto graph = n == 0 ? graph_base : std::make_shared<CLDNNGraph>(graph_base, n);
        m_graphs.push_back(graph);
    }
}

InferRequestInternal::Ptr CLDNNExecNetwork::CreateInferRequestImpl(InputsDataMap networkInputs,
                                                                   OutputsDataMap networkOutputs) {
    if (m_graphs.empty()) {
        THROW_IE_EXCEPTION << NETWORK_NOT_LOADED_str;
    }

    for (auto& graph : m_graphs) {
        if (graph == nullptr) {
            THROW_IE_EXCEPTION << NETWORK_NOT_LOADED_str;
        }

        if (!graph->IsLoaded()) {
            THROW_IE_EXCEPTION << NETWORK_NOT_LOADED_str << ": no networks created";
        }
    }

    auto ptr = std::make_shared<CLDNNInferRequest>(networkInputs, networkOutputs,
                                                   std::static_pointer_cast<CLDNNExecNetwork>(shared_from_this()));
    if (m_config.throughput_streams > 1) {
        ptr->EnableStreams();
    }
    if (m_config.useProfiling)
        ptr->EnableProfiling();
    ptr->SetGraph(m_graphs.front());

    return ptr;
}

void CLDNNExecNetwork::CreateInferRequest(IInferRequest::Ptr &asyncRequest) {
    auto syncRequestImpl = this->CreateInferRequestImpl(_networkInputs, _networkOutputs);
    syncRequestImpl->setPointerToExecutableNetworkInternal(shared_from_this());

    auto asyncTreadSafeImpl = std::make_shared<CLDNNAsyncInferRequest>(syncRequestImpl, _taskExecutor, _callbackExecutor);

    asyncRequest.reset(new InferRequestBase<CLDNNAsyncInferRequest>(asyncTreadSafeImpl), [](IInferRequest *p) { p->Release(); });
    asyncTreadSafeImpl->SetPointerToPublicInterface(asyncRequest);
}

void CLDNNExecNetwork::GetExecGraphInfo(InferenceEngine::ICNNNetwork::Ptr &graphPtr) {
    if (m_graphs.empty())
        THROW_IE_EXCEPTION << NETWORK_NOT_LOADED_str;

    m_graphs.front()->GetExecGraphInfo(graphPtr);
}

void CLDNNExecNetwork::GetConfig(const std::string &name, InferenceEngine::Parameter &result, InferenceEngine::ResponseDesc *resp) const {
    auto option = m_config.key_config_map.find(name);
    if (option != m_config.key_config_map.end()) {
        result = option->second;
    } else {
        THROW_IE_EXCEPTION << "Unsupported ExecutableNetwork config key: " << name;
    }
}

void CLDNNExecNetwork::GetMetric(const std::string &name, InferenceEngine::Parameter &result, InferenceEngine::ResponseDesc *resp) const {
    if (name == METRIC_KEY(NETWORK_NAME)) {
        IE_ASSERT(!m_graphs.empty());
        result = IE_SET_METRIC(NETWORK_NAME, m_graphs[0]->getName());
    } else if (name == METRIC_KEY(SUPPORTED_METRICS)) {
        std::vector<std::string> metrics;
        metrics.push_back(METRIC_KEY(NETWORK_NAME));
        metrics.push_back(METRIC_KEY(SUPPORTED_METRICS));
        metrics.push_back(METRIC_KEY(SUPPORTED_CONFIG_KEYS));
        metrics.push_back(METRIC_KEY(OPTIMAL_NUMBER_OF_INFER_REQUESTS));
        result = IE_SET_METRIC(SUPPORTED_METRICS, metrics);
    } else if (name == METRIC_KEY(SUPPORTED_CONFIG_KEYS)) {
        std::vector<std::string> configKeys;
        for (auto && value : m_config.key_config_map)
            configKeys.push_back(value.first);
        result = IE_SET_METRIC(SUPPORTED_CONFIG_KEYS, configKeys);
    } else if (name == METRIC_KEY(OPTIMAL_NUMBER_OF_INFER_REQUESTS)) {
        unsigned int nr = m_config.throughput_streams * 2u;
        result = IE_SET_METRIC(OPTIMAL_NUMBER_OF_INFER_REQUESTS, nr);
    } else {
        THROW_IE_EXCEPTION << "Unsupported ExecutableNetwork metric: " << name;
    }
}

void CLDNNExecNetwork::GetContext(RemoteContext::Ptr &pContext, ResponseDesc *resp) const {
    pContext = m_context;
}

};  // namespace CLDNNPlugin
