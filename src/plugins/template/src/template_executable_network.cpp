// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "template_executable_network.hpp"

#include <ie_metric_helpers.hpp>
#include <ie_plugin_config.hpp>
#include <threading/ie_executor_manager.hpp>

#include "cpp/ie_cnn_network.h"
#include "ie_icnn_network.hpp"
#include "ie_icore.hpp"
#include "ie_ngraph_utils.hpp"
#include "openvino/core/except.hpp"
#include "openvino/pass/serialize.hpp"
#include "template/template_config.hpp"
#include "template_itt.hpp"
#include "template_plugin.hpp"
#include "transformations/utils/utils.hpp"

using namespace TemplatePlugin;

// ! [executable_network:ctor_cnnnetwork]
TemplatePlugin::ExecutableNetwork::ExecutableNetwork(const std::shared_ptr<const ngraph::Function>& function,
                                                     const InferenceEngine::InputsDataMap& inputInfoMap,
                                                     const InferenceEngine::OutputsDataMap& outputsInfoMap,
                                                     const Configuration& cfg,
                                                     const Plugin::Ptr& plugin)
    : InferenceEngine::ExecutableNetworkThreadSafeDefault(nullptr, nullptr),  // Disable default threads creation
      _cfg(cfg),
      _plugin(plugin) {
    // TODO: if your plugin supports device ID (more that single instance of device can be on host machine)
    // you should select proper device based on KEY_DEVICE_ID or automatic behavior
    // In this case, _waitExecutor should also be created per device.
    try {
        CompileNetwork(function, inputInfoMap, outputsInfoMap);
        InitExecutor();  // creates thread-based executor using for async requests
    } catch (const InferenceEngine::Exception&) {
        throw;
    } catch (const std::exception& e) {
        IE_THROW(Unexpected) << "Standard exception from compilation library: " << e.what();
    } catch (...) {
        IE_THROW(Unexpected) << "Generic exception is thrown";
    }
}
// ! [executable_network:ctor_cnnnetwork]

// ! [executable_network:ctor_import_stream]
TemplatePlugin::ExecutableNetwork::ExecutableNetwork(std::istream& model,
                                                     const Configuration& cfg,
                                                     const Plugin::Ptr& plugin)
    : _cfg(cfg),
      _plugin(plugin) {
    // read XML content
    std::string xmlString;
    std::uint64_t dataSize = 0;
    model.read(reinterpret_cast<char*>(&dataSize), sizeof(dataSize));
    xmlString.resize(dataSize);
    model.read(const_cast<char*>(xmlString.c_str()), dataSize);

    // read blob content
    InferenceEngine::Blob::Ptr dataBlob;
    model.read(reinterpret_cast<char*>(&dataSize), sizeof(dataSize));
    if (0 != dataSize) {
        dataBlob = InferenceEngine::make_shared_blob<std::uint8_t>(
            InferenceEngine::TensorDesc(InferenceEngine::Precision::U8,
                                        {static_cast<std::size_t>(dataSize)},
                                        InferenceEngine::Layout::C));
        dataBlob->allocate();
        model.read(dataBlob->buffer(), dataSize);
    }

    auto cnnnetwork = _plugin->GetCore()->ReadNetwork(xmlString, std::move(dataBlob));

    // TODO: implement Import / Export of configuration options and merge with `cfg`
    // TODO: implement Import / Export of network precisions, layouts, preprocessing info
    InferenceEngine::InputsDataMap inputInfoMap = cnnnetwork.getInputsInfo();
    InferenceEngine::OutputsDataMap outputInfoMap = cnnnetwork.getOutputsInfo();

    setNetworkInputs(inputInfoMap);
    setNetworkOutputs(outputInfoMap);
    SetPointerToPlugin(_plugin->shared_from_this());

    try {
        // TODO: remove compilation, network is already compiled and serialized in compiled form
        CompileNetwork(cnnnetwork.getFunction(), inputInfoMap, outputInfoMap);
        InitExecutor();  // creates thread-based executor using for async requests
    } catch (const InferenceEngine::Exception&) {
        throw;
    } catch (const std::exception& e) {
        IE_THROW(Unexpected) << "Standard exception from compilation library: " << e.what();
    } catch (...) {
        IE_THROW(Unexpected) << "Generic exception is thrown";
    }
}
// ! [executable_network:ctor_import_stream]

// ! [executable_network:map_graph]
// forward declaration
void TransformNetwork(std::shared_ptr<ngraph::Function>& function,
                      const InferenceEngine::InputsDataMap& inputInfoMap,
                      const InferenceEngine::OutputsDataMap& outputsInfoMap);

void TemplatePlugin::ExecutableNetwork::CompileNetwork(const std::shared_ptr<const ngraph::Function>& function,
                                                       const InferenceEngine::InputsDataMap& inputInfoMap,
                                                       const InferenceEngine::OutputsDataMap& outputsInfoMap) {
    // TODO: perform actual graph compilation / mapping to backend graph representation / kernels

    // clone network
    _function = ngraph::clone_function(*function);

    // apply plugins transformations
    TransformNetwork(_function, inputInfoMap, outputsInfoMap);

    // Generate backend specific blob mappings. For example Inference Engine uses not ngraph::Result nodes friendly name
    // as inference request output names but the name of the layer before.
    size_t idx = 0;
    for (auto&& result : _function->get_results()) {
        const auto& input = result->input_value(0);
        auto name = ngraph::op::util::get_ie_output_name(input);
        if (_outputIndex.emplace(name, idx).second)
            idx++;
    }
    for (auto&& parameter : _function->get_parameters()) {
        _inputIndex.emplace(parameter->get_friendly_name(), _function->get_parameter_index(parameter));
    }

    // Perform any other steps like allocation and filling backend specific memory handles and so on
}
// ! [executable_network:map_graph]

// ! [executable_network:init_executor]
void TemplatePlugin::ExecutableNetwork::InitExecutor() {
    // Default multi-threaded configuration is balanced for throughtput and latency cases and takes into account
    // real hardware cores and NUMA nodes.
    auto streamsExecutorConfig =
        InferenceEngine::IStreamsExecutor::Config::MakeDefaultMultiThreaded(_cfg._streamsExecutorConfig);
    streamsExecutorConfig._name = "TemplateStreamsExecutor";
    // As Inference Engine CPU Streams Executor creates some additional therads
    // it is better to avoid threads recreateion as some OSs memory allocator can not manage such usage cases
    // and memory consumption can be larger than it is expected.
    // So Inference Engone provides executors cache.
    _taskExecutor = _plugin->executorManager()->getIdleCPUStreamsExecutor(streamsExecutorConfig);
    // NOTE: callback Executor is not configured. So callback will be called in the thread of the last stage of
    // inference request pipeline _callbackExecutor =
    // _plugin->executorManager()->getIdleCPUStreamsExecutor({"TemplateCallbackExecutor"});
}
// ! [executable_network:init_executor]

// ! [executable_network:create_infer_request_impl]
InferenceEngine::IInferRequestInternal::Ptr TemplatePlugin::ExecutableNetwork::CreateInferRequestImpl(
    InferenceEngine::InputsDataMap networkInputs,
    InferenceEngine::OutputsDataMap networkOutputs) {
    return std::make_shared<TemplateInferRequest>(networkInputs,
                                                  networkOutputs,
                                                  std::static_pointer_cast<ExecutableNetwork>(shared_from_this()));
}

InferenceEngine::IInferRequestInternal::Ptr TemplatePlugin::ExecutableNetwork::CreateInferRequestImpl(
    const std::vector<std::shared_ptr<const ov::Node>>& inputs,
    const std::vector<std::shared_ptr<const ov::Node>>& outputs) {
    return std::make_shared<TemplateInferRequest>(inputs,
                                                  outputs,
                                                  std::static_pointer_cast<ExecutableNetwork>(shared_from_this()));
}
// ! [executable_network:create_infer_request_impl]

// ! [executable_network:create_infer_request]
InferenceEngine::IInferRequestInternal::Ptr TemplatePlugin::ExecutableNetwork::CreateInferRequest() {
    InferenceEngine::IInferRequestInternal::Ptr internalRequest;
    if (this->_plugin && _plugin->IsNewAPI()) {
        internalRequest = CreateInferRequestImpl(_parameters, _results);
    }
    if (!internalRequest)
        internalRequest = CreateInferRequestImpl(_networkInputs, _networkOutputs);
    return std::make_shared<TemplateAsyncInferRequest>(std::static_pointer_cast<TemplateInferRequest>(internalRequest),
                                                       _taskExecutor,
                                                       _plugin->_waitExecutor,
                                                       _callbackExecutor);
}
// ! [executable_network:create_infer_request]

// ! [executable_network:get_config]
InferenceEngine::Parameter TemplatePlugin::ExecutableNetwork::GetConfig(const std::string& name) const {
    return _cfg.Get(name);
}
// ! [executable_network:get_config]

// ! [executable_network:get_metric]
InferenceEngine::Parameter TemplatePlugin::ExecutableNetwork::GetMetric(const std::string& name) const {
    // TODO: return more supported values for metrics
    if (EXEC_NETWORK_METRIC_KEY(SUPPORTED_METRICS) == name) {
        IE_SET_METRIC_RETURN(SUPPORTED_METRICS,
                             std::vector<std::string>{METRIC_KEY(NETWORK_NAME),
                                                      METRIC_KEY(SUPPORTED_METRICS),
                                                      METRIC_KEY(SUPPORTED_CONFIG_KEYS),
                                                      METRIC_KEY(OPTIMAL_NUMBER_OF_INFER_REQUESTS)});
    } else if (EXEC_NETWORK_METRIC_KEY(SUPPORTED_CONFIG_KEYS) == name) {
        std::vector<std::string> configKeys = {CONFIG_KEY(DEVICE_ID),
                                               CONFIG_KEY(PERF_COUNT),
                                               TEMPLATE_CONFIG_KEY(THROUGHPUT_STREAMS)};
        auto streamExecutorConfigKeys = InferenceEngine::IStreamsExecutor::Config{}.SupportedKeys();
        for (auto&& configKey : streamExecutorConfigKeys) {
            configKeys.emplace_back(configKey);
        }
        IE_SET_METRIC_RETURN(SUPPORTED_CONFIG_KEYS, configKeys);
    } else if (EXEC_NETWORK_METRIC_KEY(NETWORK_NAME) == name) {
        auto networkName = _function->get_friendly_name();
        IE_SET_METRIC_RETURN(NETWORK_NAME, networkName);
    } else if (EXEC_NETWORK_METRIC_KEY(OPTIMAL_NUMBER_OF_INFER_REQUESTS) == name) {
        unsigned int value = _cfg._streamsExecutorConfig._streams;
        IE_SET_METRIC_RETURN(OPTIMAL_NUMBER_OF_INFER_REQUESTS, value);
    } else {
        IE_THROW() << "Unsupported ExecutableNetwork metric: " << name;
    }
}
// ! [executable_network:get_metric]

// ! [executable_network:export]
void TemplatePlugin::ExecutableNetwork::Export(std::ostream& modelStream) {
    OV_ITT_SCOPED_TASK(itt::domains::TemplatePlugin, "ExecutableNetwork::Export");

    // Note: custom ngraph extensions are not supported
    std::map<std::string, ngraph::OpSet> custom_opsets;
    std::stringstream xmlFile, binFile;
    OPENVINO_SUPPRESS_DEPRECATED_START
    ov::pass::Serialize serializer(xmlFile, binFile, custom_opsets);
    OPENVINO_SUPPRESS_DEPRECATED_END
    serializer.run_on_model(_function);

    auto m_constants = binFile.str();
    auto m_model = xmlFile.str();

    auto dataSize = static_cast<std::uint64_t>(m_model.size());
    modelStream.write(reinterpret_cast<char*>(&dataSize), sizeof(dataSize));
    modelStream.write(m_model.c_str(), dataSize);

    dataSize = static_cast<std::uint64_t>(m_constants.size());
    modelStream.write(reinterpret_cast<char*>(&dataSize), sizeof(dataSize));
    modelStream.write(reinterpret_cast<char*>(&m_constants[0]), dataSize);

    // TODO: implement network precision, layout, preprocessing info serialization
}
// ! [executable_network:export]
