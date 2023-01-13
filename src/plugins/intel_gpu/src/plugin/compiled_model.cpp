// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ie_metric_helpers.hpp"
#include "intel_gpu/graph/serialization/binary_buffer.hpp"
#include "intel_gpu/graph/serialization/layout_serializer.hpp"
#include "intel_gpu/graph/serialization/string_serializer.hpp"
#include "intel_gpu/plugin/graph.hpp"
#include "intel_gpu/plugin/itt.hpp"
#include "intel_gpu/plugin/infer_request.hpp"
#include "intel_gpu/plugin/compiled_model.hpp"
#include "intel_gpu/plugin/async_infer_request.hpp"
#include "intel_gpu/plugin/async_infer_request_legacy.hpp"
#include "openvino/runtime/intel_gpu/properties.hpp"

#include <description_buffer.hpp>
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

CompiledModel::CompiledModel(InferenceEngine::CNNNetwork &network, std::shared_ptr<InferenceEngine::RemoteContext> context, Config config) :
    InferenceEngine::ExecutableNetworkThreadSafeDefault{[&]() -> InferenceEngine::ITaskExecutor::Ptr {
        if (config.exclusiveAsyncRequests) {
            //exclusiveAsyncRequests essentially disables the streams (and hence should be checked first) => aligned with the CPU behavior
            return executorManager()->getExecutor("GPU");
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
    m_waitExecutor(executorManager()->getIdleCPUStreamsExecutor({ "GPUWaitExecutor" })) {
    auto casted_context = std::dynamic_pointer_cast<gpu::ClContext>(context);

    OPENVINO_ASSERT((casted_context != nullptr), "Invalid remote context");

    m_context = casted_context;

    auto graph_base = std::make_shared<Graph>(network, m_context, m_config, 0);
    for (uint16_t n = 0; n < m_config.throughput_streams; n++) {
        auto graph = n == 0 ? graph_base : std::make_shared<Graph>(graph_base, n);
        m_graphs.push_back(graph);
    }
}

CompiledModel::CompiledModel(std::istream& networkModel, std::shared_ptr<InferenceEngine::RemoteContext> context, Config config) :
    InferenceEngine::ExecutableNetworkThreadSafeDefault{[&]() -> InferenceEngine::ITaskExecutor::Ptr {
        if (config.exclusiveAsyncRequests) {
            //exclusiveAsyncRequests essentially disables the streams (and hence should be checked first) => aligned with the CPU behavior
            return executorManager()->getExecutor("GPU");
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
    m_waitExecutor(executorManager()->getIdleCPUStreamsExecutor({ "GPUWaitExecutor" })) {
    auto casted_context = std::dynamic_pointer_cast<gpu::ClContext>(context);

    OPENVINO_ASSERT((casted_context != nullptr), "Invalid remote context");

    m_context = casted_context;

    cldnn::BinaryInputBuffer ib(networkModel, *getContextImpl(m_context)->GetEngine());

    // InputsInfo and OutputsInfor for CNNNetwork
    {
        size_t inputSize;
        ib >> inputSize;

        InputsDataMap inputs;

        for (size_t idx = 0; idx < inputSize; ++idx) {
            std::string name;
            std::string precision;
            std::string layout;
            ib >> name;
            ib >> precision;
            ib >> layout;

            DataPtr input = std::make_shared<Data>(name, Precision::FromStr(precision), cldnn::layout_from_string(layout));
            InputInfo::Ptr infoNew = std::make_shared<InputInfo>();
            infoNew->setInputData(input);
            inputs.emplace(std::make_pair(name, infoNew));
        }

        size_t outputSize;
        ib >> outputSize;

        OutputsDataMap outputs;

        for (size_t idx = 0; idx < outputSize; ++idx) {
            std::string name;
            std::string precision;
            std::string layout;
            ib >> name;
            ib >> precision;
            ib >> layout;

            DataPtr output = std::make_shared<Data>(name, Precision::FromStr(precision), cldnn::layout_from_string(layout));
            outputs.emplace(std::make_pair(name, output));
        }

        setNetworkInputs(inputs);
        setNetworkOutputs(outputs);
    }

    {
        std::vector<std::shared_ptr<const ov::Node>> new_params;
        size_t num_params;
        ib >> num_params;

        for (size_t idx = 0; idx < num_params; ++idx) {
            std::string param_name;
            ib >> param_name;
            ov::element::Type param_element_type;
            std::string str_element_type;
            ib >> str_element_type;
            std::stringstream oss(str_element_type);
            oss >> param_element_type;
            ov::Shape param_shape;
            size_t shape_size;
            ib >> shape_size;
            param_shape.resize(shape_size);
            for (size_t i = 0; i < shape_size; ++i) {
                size_t dim;
                ib >> dim;
                param_shape[i] = dim;
            }
            std::string str_layout;
            ib >> str_layout;
            ov::Layout param_layout(str_layout);
            std::unordered_set<std::string> param_names;
            size_t num_names;
            ib >> num_names;
            for (size_t i = 0; i < num_names; ++i) {
                std::string name;
                ib >> name;
                param_names.emplace(name);
            }

            auto new_param = std::make_shared<ov::op::v0::Parameter>(param_element_type, param_shape);
            new_param->set_friendly_name(param_name);
            new_param->set_element_type(param_element_type);
            new_param->set_layout(param_layout);
            new_param->output(0).get_tensor().set_names(param_names);
            new_param->validate_and_infer_types();
            new_params.emplace_back(new_param);
        }

        setInputs(new_params);
    }

    {
        std::vector<std::shared_ptr<const ov::Node>> new_results;
        size_t num_results;
        ib >> num_results;

        for (size_t idx = 0; idx < num_results; ++idx) {
            ov::element::Type fake_element_type;
            std::string str_element_type;
            ib >> str_element_type;
            std::stringstream oss(str_element_type);
            oss >> fake_element_type;

            ov::Shape fake_shape;
            size_t shape_size;
            ib >> shape_size;
            fake_shape.resize(shape_size);
            for (size_t i = 0; i < shape_size; ++i) {
                size_t dim;
                ib >> dim;
                fake_shape[i] = dim;
            }

            std::string fake_name;
            ib >> fake_name;

            std::string param_name;
            ib >> param_name;

            std::string str_layout;
            ib >> str_layout;
            ov::Layout param_layout(str_layout);

            std::unordered_set<std::string> param_names;
            size_t num_names;
            ib >> num_names;
            for (size_t i = 0; i < num_names; ++i) {
                std::string name;
                ib >> name;
                param_names.emplace(name);
            }

            auto fake_param = std::make_shared<ov::op::v0::Parameter>(fake_element_type, fake_shape);
            fake_param->set_friendly_name(fake_name);
            fake_param->validate_and_infer_types();

            auto new_result = std::make_shared<ov::op::v0::Result>(fake_param);
            new_result->set_friendly_name(param_name);
            new_result->set_layout(param_layout);
            new_result->output(0).get_tensor().set_names(param_names);
            new_result->validate_and_infer_types();
            new_results.emplace_back(new_result);
        }

        setOutputs(new_results);
    }

    auto graph_base = std::make_shared<Graph>(ib, m_context, m_config, 0);
    for (uint16_t n = 0; n < m_config.throughput_streams; n++) {
        auto graph = n == 0 ? graph_base : std::make_shared<Graph>(graph_base, n);
        m_graphs.push_back(graph);
    }
}

template <class T>
IInferRequestInternal::Ptr CompiledModel::GetInferRequestImpl(const std::vector<std::shared_ptr<const ov::Node>>& inputs,
                                                              const std::vector<std::shared_ptr<const ov::Node>>& outputs) {
    auto ptr = std::make_shared<T>(inputs, outputs, std::static_pointer_cast<CompiledModel>(shared_from_this()));
    if (m_config.throughput_streams > 1)
        ptr->EnableStreams();
    if (m_config.useProfiling)
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
    if (m_config.throughput_streams > 1)
        ptr->EnableStreams();
    if (m_config.useProfiling)
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

bool CompiledModel::is_serializable() {
    // Model with multiple graphs is not yet supported.
    if (m_graphs.size() != 1)
        return false;

    // Dynamic model serialization is not yet supported.
    if (m_graphs[0]->GetNetwork()->is_dynamic())
        return false;

    return true;
}

// Cache blob format:
//     [ ConstInputsDataMap / ConstOutputsDataMap ]
//     [ ov::Node::Input/ ov::Node::Output ]
//     [ ov::intel_gpu::Graph ]
void CompiledModel::Export(std::ostream& networkModel) {
    OV_ITT_SCOPED_TASK(itt::domains::intel_gpu_plugin, "CompiledModel::Export");
    if (m_graphs.empty())
        IE_THROW(NetworkNotLoaded);

    if (!is_serializable())
        return;

    cldnn::BinaryOutputBuffer ob(networkModel);

    // InputsInfo and OutputsInfo for CNNNetwork
    {
        ob << GetInputsInfo().size();

        for (const auto & in : GetInputsInfo()) {
            ob << in.first;
            std::string precision(in.second->getPrecision().name());
            ob << precision;
            std::stringstream ss;
            ss << in.second->getInputData()->getLayout();
            ob << ss.str();
        }

        ob << GetOutputsInfo().size();

        for (const auto & out : GetOutputsInfo()) {
            ob << out.first;
            std::string precision(out.second->getPrecision().name());
            ob << precision;
            std::stringstream ss;
            ss << out.second->getLayout();
            ob << ss.str();
        }
    }

    // Inputs
    {
        std::vector<std::shared_ptr<const ov::Node>> const_params = getInputs();
        ob << const_params.size();

        for (const auto& param : const_params) {
            auto new_param = ov::as_type_ptr<const ov::op::v0::Parameter>(param);
            std::string param_name = new_param->get_friendly_name();
            ov::element::Type param_element_type = new_param->get_element_type();
            ov::PartialShape param_shape = new_param->get_partial_shape();
            ov::Layout param_layout = new_param->get_layout();
            // ov::RTMap param_rt_info = new_param->output(0).get_rt_info();
            auto param_names = new_param->output(0).get_tensor().get_names();

            ob << param_name;
            std::stringstream ss;
            ss << param_element_type;
            ob << ss.str();
            ov::Shape static_shape = param_shape.get_shape();
            ob << static_shape.size();
            for (size_t dim : static_shape) {
                ob << dim;
            }
            ob << param_layout.to_string();
            ob << param_names.size();
            for (auto name : param_names) {
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
            ov::PartialShape fake_shape = new_param->get_input_partial_shape(0);
            std::string fake_name = new_param->get_input_node_ptr(0)->get_friendly_name();

            std::string param_name = new_param->get_friendly_name();
            ov::Layout param_layout = new_param->get_layout();
            auto param_names = new_param->output(0).get_tensor().get_names();

            std::stringstream ss;
            ss << fake_element_type;
            ob << ss.str();
            ov::Shape static_shape = fake_shape.get_shape();
            ob << static_shape.size();
            for (size_t dim : static_shape) {
                ob << dim;
            }
            ob << fake_name;
            ob << param_name;
            ob << param_layout.to_string();
            ob << param_names.size();
            for (auto name : param_names) {
                ob << name;
            }
        }
    }

    return m_graphs.front()->Export(ob);
}

std::shared_ptr<ngraph::Function> CompiledModel::GetExecGraphInfo() {
    if (m_graphs.empty())
        IE_THROW(NetworkNotLoaded);

    return m_graphs.front()->GetExecGraphInfo();
}

InferenceEngine::Parameter CompiledModel::GetConfig(const std::string &name) const {
    const bool is_new_api = _plugin->IsNewAPI();
    auto it = m_config.key_config_map.find(name);
    if (it != m_config.key_config_map.end()) {
        std::string val = it->second;
        if (is_new_api) {
            if (name == ov::enable_profiling) {
                return val == PluginConfigParams::YES ? true : false;
            } else if (name == ov::hint::model_priority) {
                return ov::util::from_string(val, ov::hint::model_priority);
            } else if (name == ov::intel_gpu::hint::host_task_priority) {
                return ov::util::from_string(val, ov::intel_gpu::hint::host_task_priority);
            } else if (name == ov::intel_gpu::hint::queue_priority) {
                return ov::util::from_string(val, ov::intel_gpu::hint::queue_priority);
            } else if (name == ov::intel_gpu::hint::queue_throttle) {
                return ov::util::from_string(val, ov::intel_gpu::hint::queue_throttle);
            } else if (name == ov::intel_gpu::enable_loop_unrolling) {
                return val == PluginConfigParams::YES ? true : false;
            } else if (name == ov::cache_dir) {
                return ov::util::from_string(val, ov::cache_dir);
            } else if (name == ov::hint::performance_mode) {
                return ov::util::from_string(val, ov::hint::performance_mode);
            } else if (name == ov::compilation_num_threads) {
                return ov::util::from_string(val, ov::compilation_num_threads);
            } else if (name == ov::num_streams) {
                return ov::util::from_string(val, ov::num_streams);
            } else if (name == ov::hint::num_requests) {
                return ov::util::from_string(val, ov::hint::num_requests);
            } else if (name == ov::hint::inference_precision) {
                return ov::util::from_string(val, ov::hint::inference_precision);
            } else if (name == ov::device::id) {
                return ov::util::from_string(val, ov::device::id);
            } else {
                return val;
            }
        } else {
            if (name == PluginConfigParams::KEY_MODEL_PRIORITY ||
                name == GPUConfigParams::KEY_GPU_HOST_TASK_PRIORITY)
                return Config::ConvertPropertyToLegacy(name, val);
            else
                return val;
        }
    } else {
        IE_THROW() << "Unsupported ExecutableNetwork config key: " << name;
    }
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
            ov::PropertyName{ov::cache_dir.name(), PropertyMutability::RO},
            ov::PropertyName{ov::hint::performance_mode.name(), PropertyMutability::RO},
            ov::PropertyName{ov::compilation_num_threads.name(), PropertyMutability::RO},
            ov::PropertyName{ov::num_streams.name(), PropertyMutability::RO},
            ov::PropertyName{ov::hint::num_requests.name(), PropertyMutability::RO},
            ov::PropertyName{ov::hint::inference_precision.name(), PropertyMutability::RO},
            ov::PropertyName{ov::device::id.name(), PropertyMutability::RO}
        };
    } else if (name == ov::model_name) {
        IE_ASSERT(!m_graphs.empty());
        return decltype(ov::model_name)::value_type {m_graphs[0]->getName()};
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
            if (!Config::isNewApiProperty(value.first))
                configKeys.push_back(value.first);
        IE_SET_METRIC_RETURN(SUPPORTED_CONFIG_KEYS, configKeys);
    } else if (name == ov::optimal_number_of_infer_requests) {
        unsigned int nr = m_config.throughput_streams;
        if (m_config.perfHintsConfig.ovPerfHint != CONFIG_VALUE(LATENCY))
            nr *= 2;
        return decltype(ov::optimal_number_of_infer_requests)::value_type {nr};
    } else {
        IE_THROW() << "Unsupported ExecutableNetwork metric: " << name;
    }
}

std::shared_ptr<InferenceEngine::RemoteContext> CompiledModel::GetContext() const {
    return m_context;
}

}  // namespace intel_gpu
}  // namespace ov
