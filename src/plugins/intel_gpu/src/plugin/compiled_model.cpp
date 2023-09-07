// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/plugin/legacy_api_helper.hpp"

#include "openvino/pass/serialize.hpp"
#include "openvino/runtime/intel_gpu/properties.hpp"

#include "intel_gpu/graph/serialization/binary_buffer.hpp"
#include "intel_gpu/graph/serialization/layout_serializer.hpp"
#include "intel_gpu/graph/serialization/string_serializer.hpp"
#include "intel_gpu/graph/serialization/utils.hpp"
#include "intel_gpu/graph/serialization/vector_serializer.hpp"
#include "intel_gpu/plugin/graph.hpp"
#include "intel_gpu/runtime/itt.hpp"
#include "intel_gpu/plugin/infer_request.hpp"
#include "intel_gpu/plugin/compiled_model.hpp"
#include "intel_gpu/plugin/async_infer_request.hpp"
#include "intel_gpu/plugin/async_infer_request_legacy.hpp"

#include <threading/ie_executor_manager.hpp>
#include "threading/ie_cpu_streams_executor.hpp"
#include "cpp_interfaces/interface/ie_internal_plugin_config.hpp"
#include "cpp_interfaces/interface/ie_iinfer_request_internal.hpp"
#include "ie_icore.hpp"

#include <fstream>
#include <utility>
#include <sys/types.h>
#include <chrono>
#include <cmath>
#include <algorithm>

#include <openvino/util/common_util.hpp>

using namespace InferenceEngine;
using namespace InferenceEngine::details;

namespace ov {
namespace intel_gpu {

CompiledModel::CompiledModel(InferenceEngine::CNNNetwork &network,
                             InferenceEngine::RemoteContext::Ptr context,
                             const ExecutionConfig& config,
                             InferenceEngine::InputsDataMap* inputs,
                             InferenceEngine::OutputsDataMap* outputs) :
    InferenceEngine::ExecutableNetworkThreadSafeDefault{[&]() -> InferenceEngine::ITaskExecutor::Ptr {
        if (config.get_property(ov::internal::exclusive_async_requests)) {
            //exclusiveAsyncRequests essentially disables the streams (and hence should be checked first) => aligned with the CPU behavior
            return executorManager()->getExecutor("GPU");
        }  else if (config.get_property(ov::num_streams) > 1) {
            return std::make_shared<InferenceEngine::CPUStreamsExecutor>(
                IStreamsExecutor::Config{"Intel GPU plugin executor", config.get_property(ov::num_streams)});
        } else {
            return std::make_shared<InferenceEngine::CPUStreamsExecutor>(
                IStreamsExecutor::Config{"Intel GPU plugin executor", 1});
        }
    }()},
    m_context(context),
    m_config(config),
    m_taskExecutor{ _taskExecutor },
    m_waitExecutor(executorManager()->getIdleCPUStreamsExecutor({ "GPUWaitExecutor" })),
    m_network(network) {
    auto graph_base = std::make_shared<Graph>(network, get_context_impl(m_context), m_config, 0, inputs, outputs);
    for (uint16_t n = 0; n < m_config.get_property(ov::num_streams); n++) {
        auto graph = n == 0 ? graph_base : std::make_shared<Graph>(graph_base, n);
        m_graphs.push_back(graph);
    }
}

CompiledModel::CompiledModel(cldnn::BinaryInputBuffer& ib,
                             InferenceEngine::RemoteContext::Ptr context,
                             const ExecutionConfig& config,
                             InferenceEngine::InputsDataMap* inputs,
                             InferenceEngine::OutputsDataMap* outputs) :
    InferenceEngine::ExecutableNetworkThreadSafeDefault{[&]() -> InferenceEngine::ITaskExecutor::Ptr {
        if (config.get_property(ov::internal::exclusive_async_requests)) {
            //exclusiveAsyncRequests essentially disables the streams (and hence should be checked first) => aligned with the CPU behavior
            return executorManager()->getExecutor("GPU");
        }  else if (config.get_property(ov::num_streams) > 1) {
            return std::make_shared<InferenceEngine::CPUStreamsExecutor>(
                IStreamsExecutor::Config{"Intel GPU plugin executor", config.get_property(ov::num_streams)});
        } else {
            return std::make_shared<InferenceEngine::CPUStreamsExecutor>(
                IStreamsExecutor::Config{"Intel GPU plugin executor", 1});
        }
    }()},
    m_context(context),
    m_config(config),
    m_taskExecutor{ _taskExecutor },
    m_waitExecutor(executorManager()->getIdleCPUStreamsExecutor({ "GPUWaitExecutor" })) {
    auto context_impl = get_context_impl(m_context);

    auto pos = ib.tellg();
    for (uint16_t n = 0; n < m_config.get_property(ov::num_streams); n++) {
        ib.seekg(pos);
        auto graph = std::make_shared<Graph>(ib, context_impl, m_config, n, inputs, outputs);
        m_graphs.push_back(graph);
    }
}

template <class T>
IInferRequestInternal::Ptr CompiledModel::GetInferRequestImpl(const std::vector<std::shared_ptr<const ov::Node>>& inputs,
                                                              const std::vector<std::shared_ptr<const ov::Node>>& outputs) {
    auto ptr = std::make_shared<T>(inputs, outputs, std::static_pointer_cast<CompiledModel>(shared_from_this()));
    if (m_config.get_property(ov::num_streams) > 1)
        ptr->EnableStreams();
    if (m_config.get_property(ov::enable_profiling))
        ptr->EnableProfiling();
    if (m_graphs.front()->use_external_queue())
        ptr->enable_external_queue();
    ptr->SetGraph(m_graphs.front());

    return ptr;
}

IInferRequestInternal::Ptr CompiledModel::CreateInferRequestImpl(InputsDataMap networkInputs,
                                                                 OutputsDataMap networkOutputs) {
    OV_ITT_SCOPED_TASK(itt::domains::intel_gpu_plugin, "CompiledModel::CreateInferRequestImpl");
    auto ptr = std::make_shared<InferRequestLegacy>(networkInputs, networkOutputs,
                                                    std::static_pointer_cast<CompiledModel>(shared_from_this()));
    if (m_config.get_property(ov::num_streams) > 1)
        ptr->EnableStreams();
    if (m_config.get_property(ov::enable_profiling))
        ptr->EnableProfiling();
    if (m_graphs.front()->use_external_queue())
        ptr->enable_external_queue();
    ptr->SetGraph(m_graphs.front());

    return ptr;
}

IInferRequestInternal::Ptr CompiledModel::CreateInferRequestImpl(const std::vector<std::shared_ptr<const ov::Node>>& inputs,
                                                                 const std::vector<std::shared_ptr<const ov::Node>>& outputs) {
    OV_ITT_SCOPED_TASK(itt::domains::intel_gpu_plugin, "CompiledModel::CreateInferRequestImpl");
    if (m_graphs.front()->GetMaxDynamicBatchSize() > 1)
        return GetInferRequestImpl<InferRequestLegacy>(inputs, outputs);
    else
        return GetInferRequestImpl<InferRequest>(inputs, outputs);
}

IInferRequestInternal::Ptr CompiledModel::CreateInferRequest() {
    OV_ITT_SCOPED_TASK(itt::domains::intel_gpu_plugin, "CompiledModel::CreateInferRequest");
    InferenceEngine::IInferRequestInternal::Ptr internalRequest;
    if (m_graphs.empty()) {
        OPENVINO_THROW("[GPU] Model not loaded");
    }

    for (auto& graph : m_graphs) {
        if (graph == nullptr) {
            OPENVINO_THROW("[GPU] Model not loaded");
        }

        if (!graph->IsLoaded()) {
            OPENVINO_THROW("[GPU] Model not loaded: no networks created");
        }
    }

    bool is_legacy = false;
    if (this->_plugin && _plugin->IsNewAPI()) {
        internalRequest = CreateInferRequestImpl(_parameters, _results);
        if (std::dynamic_pointer_cast<InferRequestLegacy>(internalRequest))
            is_legacy = true;
    }
    if (!internalRequest) {
        internalRequest = CreateInferRequestImpl(_networkInputs, _networkOutputs);
        is_legacy = true;
    }
    internalRequest->setPointerToExecutableNetworkInternal(shared_from_this());
    if (is_legacy) {
        return std::make_shared<AsyncInferRequestLegacy>(std::static_pointer_cast<InferRequestLegacy>(internalRequest),
                                                         m_taskExecutor,
                                                         m_waitExecutor,
                                                         _callbackExecutor);
    }
    return std::make_shared<AsyncInferRequest>(std::static_pointer_cast<InferRequest>(internalRequest),
                                               m_taskExecutor,
                                               m_waitExecutor,
                                               _callbackExecutor);
}

// Cache blob format:
//     [ ConstInputsDataMap / ConstOutputsDataMap ]
//     [ ov::Node::Input/ ov::Node::Output ]
//     [ ov::intel_gpu::Graph ]
void CompiledModel::Export(std::ostream& networkModel) {
    OV_ITT_SCOPED_TASK(itt::domains::intel_gpu_plugin, "CompiledModel::Export");
    if (m_graphs.empty())
        OPENVINO_THROW("[GPU] Model not loaded");

    cldnn::BinaryOutputBuffer ob(networkModel);

    // InputsInfo and OutputsInfo for CNNNetwork
    {
        ob << GetInputsInfo().size();

        for (const auto& in : GetInputsInfo()) {
            ob << in.first;
            std::string precision(in.second->getPrecision().name());
            ob << precision;
            std::stringstream ss;
            ss << in.second->getInputData()->getLayout();
            ob << ss.str();
            ob << in.second->getTensorDesc().getDims();
        }

        ob << GetOutputsInfo().size();

        for (const auto& out : GetOutputsInfo()) {
            ob << out.first;
            std::string precision(out.second->getPrecision().name());
            ob << precision;
            std::stringstream ss;
            ss << out.second->getLayout();
            ob << ss.str();
            ob << out.second->getTensorDesc().getDims();
        }
    }

    // Inputs
    {
        const std::vector<std::shared_ptr<const ov::Node>>& const_params = getInputs();
        ob << const_params.size();

        for (const auto& param : const_params) {
            auto new_param = ov::as_type_ptr<const ov::op::v0::Parameter>(param);
            ov::element::Type param_element_type = new_param->get_element_type();

            const std::string& param_name = new_param->get_friendly_name();
            const ov::PartialShape& param_shape = new_param->get_partial_shape();
            const ov::Layout& param_layout = new_param->get_layout();
            const auto& param_names = new_param->output(0).get_tensor().get_names();

            ob << param_name;
            std::stringstream ss;
            ss << param_element_type;
            ob << ss.str();
            ob << param_shape;
            ob << param_layout.to_string();
            ob << param_names.size();
            for (const auto& name : param_names) {
                ob << name;
            }
        }
    }

    // Outputs
    {
        std::vector<std::shared_ptr<const ov::Node>> const_results = getOutputs();
        ob << const_results.size();

        for (const auto& param : const_results) {
            auto new_param = ov::as_type_ptr<const ov::op::v0::Result>(param);
            ov::element::Type fake_element_type = new_param->get_input_element_type(0);

            const std::string& fake_name = new_param->get_input_node_ptr(0)->get_friendly_name();
            const std::string& param_name = new_param->get_friendly_name();
            const ov::PartialShape& fake_shape = new_param->get_input_partial_shape(0);
            const ov::Layout& param_layout = new_param->get_layout();
            const auto& param_names = new_param->output(0).get_tensor().get_names();


            std::stringstream ss;
            ss << fake_element_type;
            ob << ss.str();
            ob << fake_shape;
            ob << fake_name;
            ob << param_name;
            ob << param_layout.to_string();
            ob << param_names.size();
            for (const auto& name : param_names) {
                ob << name;
            }
        }
    }

    if (m_graphs.front()->GetNetwork()->is_dynamic()) {
        ob << true;
        ov::pass::StreamSerialize serializer(networkModel, {}, ov::pass::Serialize::Version::UNSPECIFIED);
        serializer.run_on_model(std::const_pointer_cast<ngraph::Function>(m_network.getFunction()));
    } else {
        ob << false;
        m_graphs.front()->Export(ob);
    }
}

std::shared_ptr<ngraph::Function> CompiledModel::GetExecGraphInfo() {
    if (m_graphs.empty())
        OPENVINO_THROW("[GPU] Model not loaded");

    return m_graphs.front()->GetExecGraphInfo();
}

InferenceEngine::Parameter CompiledModel::GetConfig(const std::string &name) const {
    auto actual_name = name;
    if (LegacyAPIHelper::is_legacy_property({name, nullptr}, _plugin->IsNewAPI())) {
        actual_name = LegacyAPIHelper::convert_legacy_property({name, nullptr}).first;
    }

    auto val = m_config.get_property(actual_name);
    if (LegacyAPIHelper::is_legacy_property({name, nullptr}, _plugin->IsNewAPI())) {
        val = LegacyAPIHelper::convert_to_legacy_property({actual_name, val}).second;
    }

    return val;
}

InferenceEngine::Parameter CompiledModel::GetMetric(const std::string &name) const {
    if (name == ov::supported_properties) {
        return decltype(ov::supported_properties)::value_type {
            // Metrics
            ov::PropertyName{ov::supported_properties.name(), PropertyMutability::RO},
            ov::PropertyName{ov::model_name.name(), PropertyMutability::RO},
            ov::PropertyName{ov::optimal_number_of_infer_requests.name(), PropertyMutability::RO},

            // Configs
            ov::PropertyName{ov::enable_profiling.name(), PropertyMutability::RO},
            ov::PropertyName{ov::hint::model_priority.name(), PropertyMutability::RO},
            ov::PropertyName{ov::intel_gpu::hint::host_task_priority.name(), PropertyMutability::RO},
            ov::PropertyName{ov::intel_gpu::hint::queue_priority.name(), PropertyMutability::RO},
            ov::PropertyName{ov::intel_gpu::hint::queue_throttle.name(), PropertyMutability::RO},
            ov::PropertyName{ov::intel_gpu::enable_loop_unrolling.name(), PropertyMutability::RO},
            ov::PropertyName{ov::intel_gpu::disable_winograd_convolution.name(), PropertyMutability::RO},
            ov::PropertyName{ov::cache_dir.name(), PropertyMutability::RO},
            ov::PropertyName{ov::hint::performance_mode.name(), PropertyMutability::RO},
            ov::PropertyName{ov::hint::execution_mode.name(), PropertyMutability::RO},
            ov::PropertyName{ov::compilation_num_threads.name(), PropertyMutability::RO},
            ov::PropertyName{ov::num_streams.name(), PropertyMutability::RO},
            ov::PropertyName{ov::hint::num_requests.name(), PropertyMutability::RO},
            ov::PropertyName{ov::hint::inference_precision.name(), PropertyMutability::RO},
            ov::PropertyName{ov::device::id.name(), PropertyMutability::RO},
            ov::PropertyName{ov::execution_devices.name(), PropertyMutability::RO}
        };
    } else if (name == ov::model_name) {
        OPENVINO_ASSERT(!m_graphs.empty());
        return decltype(ov::model_name)::value_type {m_graphs[0]->getName()};
    } else if (name == METRIC_KEY(SUPPORTED_METRICS)) {
        std::vector<std::string> metrics;
        metrics.push_back(METRIC_KEY(NETWORK_NAME));
        metrics.push_back(METRIC_KEY(SUPPORTED_METRICS));
        metrics.push_back(METRIC_KEY(SUPPORTED_CONFIG_KEYS));
        metrics.push_back(METRIC_KEY(OPTIMAL_NUMBER_OF_INFER_REQUESTS));
        IE_SET_METRIC_RETURN(SUPPORTED_METRICS, metrics);
    } else if (name == METRIC_KEY(SUPPORTED_CONFIG_KEYS)) {
        static const std::vector<std::string> configKeys {
            CONFIG_KEY(MODEL_PRIORITY),
            CONFIG_KEY(PERFORMANCE_HINT),
            CONFIG_KEY(PERFORMANCE_HINT_NUM_REQUESTS),
            CONFIG_KEY(PERF_COUNT),
            CONFIG_KEY(CONFIG_FILE),
            CONFIG_KEY(DEVICE_ID),
            CONFIG_KEY(EXCLUSIVE_ASYNC_REQUESTS),
            CONFIG_KEY(CACHE_DIR),
            CONFIG_KEY(GPU_THROUGHPUT_STREAMS),
            GPU_CONFIG_KEY(PLUGIN_PRIORITY),
            GPU_CONFIG_KEY(PLUGIN_THROTTLE),
            GPU_CONFIG_KEY(HOST_TASK_PRIORITY),
            GPU_CONFIG_KEY(NV12_TWO_INPUTS),
            GPU_CONFIG_KEY(MAX_NUM_THREADS),
            GPU_CONFIG_KEY(ENABLE_LOOP_UNROLLING),
        };
        IE_SET_METRIC_RETURN(SUPPORTED_CONFIG_KEYS, configKeys);
    } else if (name == ov::optimal_number_of_infer_requests) {
        unsigned int nr = m_config.get_property(ov::num_streams);
        if (m_config.get_property(ov::hint::performance_mode) != ov::hint::PerformanceMode::LATENCY)
            nr *= 2;
        return decltype(ov::optimal_number_of_infer_requests)::value_type {nr};
    } else if (name == ov::execution_devices) {
        return decltype(ov::execution_devices)::value_type{m_context->getDeviceName()};
    } else {
        OPENVINO_THROW("[GPU] Unsupported CompiledModel property: ", name);
    }
}

std::shared_ptr<InferenceEngine::RemoteContext> CompiledModel::GetContext() const {
    return m_context;
}

}  // namespace intel_gpu
}  // namespace ov
