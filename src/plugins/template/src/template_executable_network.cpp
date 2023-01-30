// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "template_executable_network.hpp"

#include <ie_metric_helpers.hpp>
#include <ie_plugin_config.hpp>
#include <threading/ie_executor_manager.hpp>

#include "converter_utils.hpp"
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

namespace {

InferenceEngine::SizeVector get_dims(const ov::Output<ov::Node>& port) {
    InferenceEngine::SizeVector dims = {};
    const auto& p_shape = port.get_partial_shape();
    if (p_shape.is_static())
        dims = p_shape.get_shape();
    return dims;
}

}  // namespace

namespace ov {
namespace legacy_convert {

void fill_input_info(const ov::Output<ov::Node>& input, InferenceEngine::InputInfo::Ptr& input_info) {
    if (!input_info) {
        // Create input info
        auto param_name = input.get_node()->get_friendly_name();
        auto dims = get_dims(input);
        InferenceEngine::TensorDesc desc(InferenceEngine::details::convertPrecision(input.get_element_type()),
                                         dims,
                                         InferenceEngine::TensorDesc::getLayoutByDims(dims));
        auto data = std::make_shared<InferenceEngine::Data>(param_name, desc);
        input_info = std::make_shared<InferenceEngine::InputInfo>();
        input_info->setInputData(data);
    }
    auto& rt_info = input.get_rt_info();
    auto it = rt_info.find("ie_legacy_preproc");
    if (it != rt_info.end()) {
        input_info->getPreProcess() = it->second.as<InferenceEngine::PreProcessInfo>();
    }
    it = rt_info.find("ie_legacy_td");
    if (it != rt_info.end()) {
        auto td = it->second.as<InferenceEngine::TensorDesc>();
        input_info->getInputData()->reshape(td.getDims(), td.getLayout());
        input_info->setPrecision(td.getPrecision());
    }
}
void fill_output_info(const ov::Output<ov::Node>& output, InferenceEngine::DataPtr& output_info) {
    if (!output_info) {
        // Create input info
        const auto& res_name = ov::op::util::create_ie_output_name(output);
        auto dims = get_dims(output);
        InferenceEngine::TensorDesc desc(InferenceEngine::details::convertPrecision(output.get_element_type()),
                                         dims,
                                         InferenceEngine::TensorDesc::getLayoutByDims(dims));
        output_info = std::make_shared<InferenceEngine::Data>(res_name, desc);
    }
    auto& rt_info = output.get_rt_info();
    auto it = rt_info.find("ie_legacy_td");
    if (it != rt_info.end()) {
        auto td = it->second.as<InferenceEngine::TensorDesc>();
        output_info->reshape(td.getDims(), td.getLayout());
        output_info->setPrecision(td.getPrecision());
    }
}
}  // namespace legacy_convert
}  // namespace ov

// ! [executable_network:ctor_cnnnetwork]
TemplatePlugin::ExecutableNetwork::ExecutableNetwork(const std::shared_ptr<const ov::Model>& model,
                                                     const Configuration& cfg,
                                                     const std::shared_ptr<const Plugin>& plugin)
    : InferenceEngine::ExecutableNetworkThreadSafeDefault(nullptr, nullptr),  // Disable default threads creation
      m_model(model->clone()),
      _cfg(cfg),
      _plugin(plugin) {
    // TODO: if your plugin supports device ID (more that single instance of device can be on host machine)
    // you should select proper device based on KEY_DEVICE_ID or automatic behavior
    // In this case, _waitExecutor should also be created per device.
    try {
        CompileNetwork(m_model);
        InitExecutor();  // creates thread-based executor using for async requests
    } catch (const InferenceEngine::Exception&) {
        throw;
    } catch (const std::exception& e) {
        IE_THROW(Unexpected) << "Standard exception from compilation library: " << e.what();
    } catch (...) {
        IE_THROW(Unexpected) << "Generic exception is thrown";
    }
    for (const auto& input : m_model->inputs()) {
        InferenceEngine::InputInfo::Ptr input_info;
        ov::legacy_convert::fill_input_info(input, input_info);
        _networkInputs[input_info->name()] = input_info;
        _parameters.emplace_back(input.get_node_shared_ptr());
    }
    for (const auto& result : m_model->get_results()) {
        const auto output = result->output(0);
        InferenceEngine::DataPtr output_info;
        ov::legacy_convert::fill_output_info(output, output_info);
        _networkOutputs[output_info->getName()] = output_info;
        _results.emplace_back(result);
    }
}
// ! [executable_network:ctor_cnnnetwork]

// ! [executable_network:ctor_import_stream]
TemplatePlugin::ExecutableNetwork::ExecutableNetwork(std::istream& model,
                                                     const Configuration& cfg,
                                                     const std::shared_ptr<const Plugin>& plugin)
    : _cfg(cfg),
      _plugin(plugin) {
    // read XML content
    std::string xmlString;
    std::uint64_t dataSize = 0;
    model.read(reinterpret_cast<char*>(&dataSize), sizeof(dataSize));
    xmlString.resize(dataSize);
    model.read(const_cast<char*>(xmlString.c_str()), dataSize);

    // read blob content
    ov::Tensor weights;
    model.read(reinterpret_cast<char*>(&dataSize), sizeof(dataSize));
    if (0 != dataSize) {
        weights = ov::Tensor(ov::element::u8, ov::Shape{dataSize});
        model.read(weights.data<char>(), dataSize);
    }

    m_model = _plugin->get_core()->read_model(xmlString, weights);

    // SetPointerToPlugin(_plugin->shared_from_this());

    try {
        // TODO: remove compilation, network is already compiled and serialized in compiled form
        CompileNetwork(m_model);
        InitExecutor();  // creates thread-based executor using for async requests
    } catch (const InferenceEngine::Exception&) {
        throw;
    } catch (const std::exception& e) {
        IE_THROW(Unexpected) << "Standard exception from compilation library: " << e.what();
    } catch (...) {
        IE_THROW(Unexpected) << "Generic exception is thrown";
    }
    for (const auto& input : m_model->inputs()) {
        InferenceEngine::InputInfo::Ptr input_info;
        ov::legacy_convert::fill_input_info(input, input_info);
        _networkInputs[input_info->name()] = input_info;
        _parameters.emplace_back(input.get_node_shared_ptr());
    }
    for (const auto& result : m_model->get_results()) {
        const auto output = result->output(0);
        InferenceEngine::DataPtr output_info;
        ov::legacy_convert::fill_output_info(output, output_info);
        _networkOutputs[output_info->getName()] = output_info;
        _results.emplace_back(result);
    }
}
// ! [executable_network:ctor_import_stream]

// ! [executable_network:map_graph]
// forward declaration
void TransformNetwork(const std::shared_ptr<ov::Model>& model);

void TemplatePlugin::ExecutableNetwork::CompileNetwork(const std::shared_ptr<ov::Model>& model) {
    // apply plugins transformations
    TransformNetwork(model);

    // Generate backend specific blob mappings. For example Inference Engine uses not ngraph::Result nodes friendly name
    // as inference request output names but the name of the layer before.
    size_t idx = 0;
    for (auto&& result : model->get_results()) {
        const auto& input = result->input_value(0);
        auto name = ov::op::util::get_ie_output_name(input);
        if (_outputIndex.emplace(name, idx).second)
            idx++;
    }
    for (auto&& parameter : model->get_parameters()) {
        _inputIndex.emplace(parameter->get_friendly_name(), model->get_parameter_index(parameter));
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
    _taskExecutor = _plugin->get_executor_manager()->getIdleCPUStreamsExecutor(streamsExecutorConfig);
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
    if (this->_plugin && _plugin->is_new_api()) {
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
        auto networkName = m_model->get_friendly_name();
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
