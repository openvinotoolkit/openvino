// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ie_metric_helpers.hpp"
#include "intel_gpu/plugin/graph.hpp"
#include "intel_gpu/plugin/itt.hpp"
#include "intel_gpu/plugin/infer_request.hpp"
#include "intel_gpu/plugin/compiled_model.hpp"
#include "intel_gpu/plugin/async_infer_request.hpp"

#include <description_buffer.hpp>
#include <threading/ie_executor_manager.hpp>
#include "threading/ie_cpu_streams_executor.hpp"
#include "cpp_interfaces/interface/ie_iinfer_request_internal.hpp"
#include "ie_icore.hpp"

#include <fstream>
#include <utility>
#include <sys/types.h>
#include <chrono>
#include <cmath>
#include <algorithm>

using namespace InferenceEngine;
using namespace InferenceEngine::details;

namespace ov {
namespace runtime {
namespace intel_gpu {

CompiledModel::CompiledModel(InferenceEngine::CNNNetwork &network, std::shared_ptr<InferenceEngine::RemoteContext> context, Config config) :
    InferenceEngine::ExecutableNetworkThreadSafeDefault{[&]() -> InferenceEngine::ITaskExecutor::Ptr {
        if (config.exclusiveAsyncRequests) {
            //exclusiveAsyncRequests essentially disables the streams (and hence should be checked first) => aligned with the CPU behavior
            return ExecutorManager::getInstance()->getExecutor("GPU");
        }  else if (config.throughput_streams > 1) {
            return std::make_shared<InferenceEngine::CPUStreamsExecutor>(
                IStreamsExecutor::Config{"Intel GPU plugin executor", config.throughput_streams});
        } else {
            return std::make_shared<InferenceEngine::CPUStreamsExecutor>(
                IStreamsExecutor::Config{"Intel GPU plugin executor", 1});
        }
    }()},
    m_config(config),
    m_taskExecutor{ _taskExecutor },
    m_waitExecutor(InferenceEngine::ExecutorManager::getInstance()->getIdleCPUStreamsExecutor({ "GPUWaitExecutor" })) {
    auto casted_context = std::dynamic_pointer_cast<gpu::ClContext>(context);

    if (nullptr == casted_context) {
        IE_THROW() << "Invalid remote context";
    }

    m_context = casted_context;

    auto graph_base = std::make_shared<Graph>(network, m_context, m_config, 0);
    for (uint16_t n = 0; n < m_config.throughput_streams; n++) {
        auto graph = n == 0 ? graph_base : std::make_shared<Graph>(graph_base, n);
        m_graphs.push_back(graph);
    }
}

IInferRequestInternal::Ptr CompiledModel::CreateInferRequestImpl(InputsDataMap networkInputs,
                                                                 OutputsDataMap networkOutputs) {
    OV_ITT_SCOPED_TASK(itt::domains::intel_gpu_plugin, "CompiledModel::CreateInferRequestImpl");
    auto ptr = std::make_shared<InferRequest>(networkInputs, networkOutputs,
                                              std::static_pointer_cast<CompiledModel>(shared_from_this()));
    if (m_config.throughput_streams > 1) {
        ptr->EnableStreams();
    }
    if (m_config.useProfiling)
        ptr->EnableProfiling();
    if (m_graphs.front()->use_external_queue()) {
        ptr->enable_external_queue();
    }
    ptr->SetGraph(m_graphs.front());

    return ptr;
}

IInferRequestInternal::Ptr CompiledModel::CreateInferRequestImpl(const std::vector<std::shared_ptr<const ov::Node>>& inputs,
                                                                 const std::vector<std::shared_ptr<const ov::Node>>& outputs) {
    OV_ITT_SCOPED_TASK(itt::domains::intel_gpu_plugin, "CompiledModel::CreateInferRequestImpl");
    auto ptr = std::make_shared<InferRequest>(inputs, outputs,
                                              std::static_pointer_cast<CompiledModel>(shared_from_this()));
    if (m_config.throughput_streams > 1) {
        ptr->EnableStreams();
    }
    if (m_config.useProfiling)
        ptr->EnableProfiling();

    if (m_graphs.front()->use_external_queue()) {
        ptr->enable_external_queue();
    }
    ptr->SetGraph(m_graphs.front());

    return ptr;
}

IInferRequestInternal::Ptr CompiledModel::CreateInferRequest() {
    OV_ITT_SCOPED_TASK(itt::domains::intel_gpu_plugin, "CompiledModel::CreateInferRequest");
    InferenceEngine::IInferRequestInternal::Ptr internalRequest;
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

    if (this->_plugin && this->_plugin->GetCore() && this->_plugin->GetCore()->isNewAPI())
        internalRequest = CreateInferRequestImpl(_parameters, _results);
    if (!internalRequest)
        internalRequest = CreateInferRequestImpl(_networkInputs, _networkOutputs);
    internalRequest->setPointerToExecutableNetworkInternal(shared_from_this());
    return std::make_shared<AsyncInferRequest>(std::static_pointer_cast<InferRequest>(internalRequest),
                                               m_taskExecutor,
                                               m_waitExecutor,
                                               _callbackExecutor);
}

std::shared_ptr<ngraph::Function> CompiledModel::GetExecGraphInfo() {
    if (m_graphs.empty())
        IE_THROW(NetworkNotLoaded);

    return m_graphs.front()->GetExecGraphInfo();
}

InferenceEngine::Parameter CompiledModel::GetConfig(const std::string &name) const {
    auto it = m_config.key_config_map.find(name);
    if (it != m_config.key_config_map.end()) {
        return it->second;
    } else {
        IE_THROW() << "Unsupported ExecutableNetwork config key: " << name;
    }
}

InferenceEngine::Parameter CompiledModel::GetMetric(const std::string &name) const {
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
        unsigned int nr = m_config.throughput_streams;
        if (m_config.perfHintsConfig.ovPerfHint != CONFIG_VALUE(LATENCY))
            nr *= 2;
        IE_SET_METRIC_RETURN(OPTIMAL_NUMBER_OF_INFER_REQUESTS, nr);
    } else {
        IE_THROW() << "Unsupported ExecutableNetwork metric: " << name;
    }
}

std::shared_ptr<InferenceEngine::RemoteContext> CompiledModel::GetContext() const {
    return m_context;
}

}  // namespace intel_gpu
}  // namespace runtime
}  // namespace ov
