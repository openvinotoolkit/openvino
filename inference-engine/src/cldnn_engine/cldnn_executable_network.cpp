// Copyright (C) 2018-2021 Intel Corporation
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
#include "cldnn_itt.h"

#include <description_buffer.hpp>
#include "cldnn_infer_request.h"
#include <threading/ie_executor_manager.hpp>
#include "cldnn_async_infer_request.h"
#include <fstream>
#include <utility>
#include <sys/types.h>

#include "cldnn_executable_network.h"
#include "threading/ie_cpu_streams_executor.hpp"
#include "cpp_interfaces/interface/ie_iinfer_request_internal.hpp"


using namespace InferenceEngine;
using namespace InferenceEngine::details;

namespace CLDNNPlugin {

CLDNNExecNetwork::CLDNNExecNetwork(InferenceEngine::CNNNetwork &network, RemoteContext::Ptr context, Config config) :
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
        IE_THROW() << "Invalid remote context";
    }

    m_context = casted_context;

    auto graph_base = std::make_shared<CLDNNGraph>(network, m_context, m_config, 0);
    for (uint16_t n = 0; n < m_config.throughput_streams; n++) {
        auto graph = n == 0 ? graph_base : std::make_shared<CLDNNGraph>(graph_base, n);
        m_graphs.push_back(graph);
    }
}

IInferRequestInternal::Ptr CLDNNExecNetwork::CreateInferRequestImpl(InputsDataMap networkInputs,
                                                                    OutputsDataMap networkOutputs) {
    OV_ITT_SCOPED_TASK(itt::domains::CLDNNPlugin, "CLDNNExecNetwork::CreateInferRequestImpl");
    if (m_graphs.empty()) {
        IE_THROW(NetworkNotLoaded);
    }

    for (auto& graph : m_graphs) {
        if (graph == nullptr) {
            IE_THROW(NetworkNotLoaded);
        }

        if (!graph->IsLoaded()) {
            IE_THROW(NetworkNotLoaded) << ": no networks created";
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

IInferRequestInternal::Ptr CLDNNExecNetwork::CreateInferRequest() {
    OV_ITT_SCOPED_TASK(itt::domains::CLDNNPlugin, "CLDNNExecNetwork::CreateInferRequest");
    return CreateAsyncInferRequestFromSync<CLDNNAsyncInferRequest>();
}

InferenceEngine::CNNNetwork CLDNNExecNetwork::GetExecGraphInfo() {
    if (m_graphs.empty())
        IE_THROW(NetworkNotLoaded);

    return m_graphs.front()->GetExecGraphInfo();
}

InferenceEngine::Parameter CLDNNExecNetwork::GetConfig(const std::string &name) const {
    auto it = m_config.key_config_map.find(name);
    if (it != m_config.key_config_map.end()) {
        return it->second;
    } else {
        IE_THROW() << "Unsupported ExecutableNetwork config key: " << name;
    }
}

InferenceEngine::Parameter CLDNNExecNetwork::GetMetric(const std::string &name) const {
    if (name == METRIC_KEY(NETWORK_NAME)) {
        IE_ASSERT(!m_graphs.empty());
        IE_SET_METRIC_RETURN(NETWORK_NAME, m_graphs[0]->getName());
    } else if (name == METRIC_KEY(SUPPORTED_METRICS)) {
        std::vector<std::string> metrics;
        metrics.push_back(METRIC_KEY(NETWORK_NAME));
        metrics.push_back(METRIC_KEY(SUPPORTED_METRICS));
        metrics.push_back(METRIC_KEY(SUPPORTED_CONFIG_KEYS));
        metrics.push_back(METRIC_KEY(OPTIMAL_NUMBER_OF_INFER_REQUESTS));
        IE_SET_METRIC_RETURN(SUPPORTED_METRICS, metrics);
    } else if (name == METRIC_KEY(SUPPORTED_CONFIG_KEYS)) {
        std::vector<std::string> configKeys;
        for (auto && value : m_config.key_config_map)
            configKeys.push_back(value.first);
        IE_SET_METRIC_RETURN(SUPPORTED_CONFIG_KEYS, configKeys);
    } else if (name == METRIC_KEY(OPTIMAL_NUMBER_OF_INFER_REQUESTS)) {
        unsigned int nr = m_config.throughput_streams * 2u;
        IE_SET_METRIC_RETURN(OPTIMAL_NUMBER_OF_INFER_REQUESTS, nr);
    } else {
        IE_THROW() << "Unsupported ExecutableNetwork metric: " << name;
    }
}

RemoteContext::Ptr CLDNNExecNetwork::GetContext() const {
    return m_context;
}

};  // namespace CLDNNPlugin
