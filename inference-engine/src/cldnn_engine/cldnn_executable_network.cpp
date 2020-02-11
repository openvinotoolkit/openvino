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
#include <graph_tools.hpp>
#include <ie_layers_internal.hpp>
#include <net_pass.h>
#include "cldnn_infer_request.h"
#include <cpp_interfaces/ie_executor_manager.hpp>
#include "details/caseless.hpp"
#include "cldnn_async_infer_request.h"
#include <fstream>
#include <utility>
#include <sys/types.h>

#include <exec_graph_info.hpp>
#include "cldnn_executable_network.h"
#include "cldnn_streams_task_executor.h"


using namespace InferenceEngine;
using namespace InferenceEngine::details;

namespace CLDNNPlugin {
unsigned int CLDNNExecNetwork::GetWaitingCounter() { return MultiWorkerTaskExecutor::GetWaitingCounter(); }
unsigned int CLDNNExecNetwork::GetRunningCounter() { return CLDNNInferRequest::GetRunningCounter(); }

CLDNNExecNetwork::CLDNNExecNetwork(InferenceEngine::ICNNNetwork &network, RemoteContext::Ptr context, Config config) : m_config(config) {
    auto casted_context = std::dynamic_pointer_cast<gpu::ClContext>(context);

    if (nullptr == casted_context) {
        THROW_IE_EXCEPTION << "Invalid remote context";
    }

    m_context = casted_context;

    // graph(s) initialization in taskExecutor threads (streams), in parallel (in case of streams)
    std::vector<InferenceEngine::Task> tasks;

    auto graph_base = std::make_shared<CLDNNGraph>(network, m_context, m_config, 0);
    for (uint16_t n = 0; n < m_config.throughput_streams; n++) {
        auto graph = n == 0 ? graph_base : std::make_shared<CLDNNGraph>(graph_base, n);
        m_graphs.push_back(graph);
        tasks.push_back([=]() {
            CLDNNPlugin::MultiWorkerTaskExecutor::ptrContext.ptrGraph = graph;
        });
    }

    if (m_config.throughput_streams > 1) {
        // special executor with as many threads as requested #streams, each with it's own initialization task
        _taskExecutor = std::make_shared<MultiWorkerTaskExecutor>(tasks);
    } else {
        if (m_config.exclusiveAsyncRequests) {
            ExecutorManager *executorManager = ExecutorManager::getInstance();
            _taskExecutor = executorManager->getExecutor("GPU");
        }
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

    auto ptr = std::make_shared<CLDNNInferRequest>(networkInputs, networkOutputs);
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
