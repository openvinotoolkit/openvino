// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "template_executable_network.hpp"

#include <ie_metric_helpers.hpp>
#include <ie_plugin_config.hpp>
#include <memory>
#include <threading/ie_executor_manager.hpp>

#include "cpp/ie_cnn_network.h"
#include "ie_common.h"
#include "ie_icnn_network.hpp"
#include "ie_icore.hpp"
#include "ie_ngraph_utils.hpp"
#include "openvino/core/except.hpp"
#include "openvino/pass/serialize.hpp"
#include "openvino/runtime/icompiled_model.hpp"
#include "openvino/runtime/iinfer_request.hpp"
#include "template/template_config.hpp"
#include "template_itt.hpp"
#include "template_plugin.hpp"
#include "transformations/utils/utils.hpp"

using namespace TemplatePlugin;

// ! [executable_network:ctor_cnnnetwork]
TemplatePlugin::ExecutableNetwork::ExecutableNetwork(const std::shared_ptr<ov::Model>& model,
                                                     const Configuration& cfg,
                                                     const std::shared_ptr<const Plugin>& plugin)
    : ov::ICompiledModel(model, plugin),  // Disable default threads creation
      _cfg(cfg) {
    // TODO: if your plugin supports device ID (more that single instance of device can be on host machine)
    // you should select proper device based on KEY_DEVICE_ID or automatic behavior
    // In this case, _waitExecutor should also be created per device.
    try {
        m_model = model->clone();
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

std::shared_ptr<const Plugin> ExecutableNetwork::get_template_plugin() const {
    auto template_plugin = std::dynamic_pointer_cast<const Plugin>(get_plugin());

    OPENVINO_ASSERT(template_plugin);
    return template_plugin;
}

// ! [executable_network:ctor_import_stream]
TemplatePlugin::ExecutableNetwork::ExecutableNetwork(std::istream& model,
                                                     const Configuration& cfg,
                                                     const std::shared_ptr<const Plugin>& plugin)
    : ov::ICompiledModel(nullptr, plugin),
      _cfg(cfg) {
    // read XML content
    std::string xmlString;
    std::uint64_t dataSize = 0;
    model.read(reinterpret_cast<char*>(&dataSize), sizeof(dataSize));
    xmlString.resize(dataSize);
    model.read(const_cast<char*>(xmlString.c_str()), dataSize);

    // read blob content
    ov::Tensor data_tensor;
    model.read(reinterpret_cast<char*>(&dataSize), sizeof(dataSize));
    if (0 != dataSize) {
        data_tensor = std::move(ov::Tensor(ov::element::i8, {dataSize}));
        model.read(data_tensor.data<char>(), dataSize);
    }

    m_model = get_template_plugin()->get_core()->read_model(xmlString, data_tensor);
    try {
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
    m_task_executor = get_template_plugin()->get_executor_manager()->getIdleCPUStreamsExecutor(streamsExecutorConfig);
    // NOTE: callback Executor is not configured. So callback will be called in the thread of the last stage of
    // inference request pipeline _callbackExecutor =
    get_template_plugin()->get_executor_manager()->getIdleCPUStreamsExecutor({"TemplateCallbackExecutor"});
}
// ! [executable_network:init_executor]

// ! [executable_network:create_infer_request_impl]
std::shared_ptr<ov::IInferRequest> TemplatePlugin::ExecutableNetwork::create_infer_request_impl() const {
    return std::make_shared<TemplateInferRequest>(std::const_pointer_cast<ExecutableNetwork>(
        std::static_pointer_cast<const ExecutableNetwork>(shared_from_this())));
}
// ! [executable_network:create_infer_request_impl]

// ! [executable_network:create_infer_request]
std::shared_ptr<ov::IInferRequest> TemplatePlugin::ExecutableNetwork::create_infer_request() const {
    std::shared_ptr<ov::IInferRequest> internalRequest;
    internalRequest = create_infer_request_impl();
    return std::make_shared<TemplateAsyncInferRequest>(std::static_pointer_cast<TemplateInferRequest>(internalRequest),
                                                       m_task_executor,
                                                       get_template_plugin()->_waitExecutor,
                                                       m_callback_executor);
}
// ! [executable_network:create_infer_request]

// ! [executable_network:get_config]
InferenceEngine::Parameter TemplatePlugin::ExecutableNetwork::get_property(const std::string& name) const {
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
        auto networkName = m_model->get_friendly_name();
        IE_SET_METRIC_RETURN(NETWORK_NAME, networkName);
    } else if (EXEC_NETWORK_METRIC_KEY(OPTIMAL_NUMBER_OF_INFER_REQUESTS) == name) {
        unsigned int value = _cfg._streamsExecutorConfig._streams;
        IE_SET_METRIC_RETURN(OPTIMAL_NUMBER_OF_INFER_REQUESTS, value);
    }
    return _cfg.Get(name);
}
// ! [executable_network:get_config]

// ! [executable_network:export]
void TemplatePlugin::ExecutableNetwork::export_model(std::ostream& modelStream) const {
    OV_ITT_SCOPED_TASK(itt::domains::TemplatePlugin, "ExecutableNetwork::Export");

    // Note: custom ngraph extensions are not supported
    std::map<std::string, ngraph::OpSet> custom_opsets;
    std::stringstream xmlFile, binFile;
    OPENVINO_SUPPRESS_DEPRECATED_START
    ov::pass::Serialize serializer(xmlFile, binFile, custom_opsets);
    OPENVINO_SUPPRESS_DEPRECATED_END
    serializer.run_on_model(m_model);

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
