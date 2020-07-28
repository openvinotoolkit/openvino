// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//


#include <utility>
#include <memory>
#include <vector>
#include <sstream>
#include <regex>
#include <string>
#include <map>

#include <ie_metric_helpers.hpp>
#include <details/ie_cnn_network_tools.h>
#include <ie_plugin_config.hpp>
#include <ie_util_internal.hpp>
#include <inference_engine.hpp>
#include <file_utils.h>
#include <cpp_interfaces/base/ie_plugin_base.hpp>
#include <cpp_interfaces/interface/ie_internal_plugin_config.hpp>
#include <threading/ie_executor_manager.hpp>
#include <graph_tools.hpp>
#include <ie_input_info.hpp>
#include <ie_layouts.h>
#include <hetero/hetero_plugin_config.hpp>
#include <template/template_config.hpp>

#include "template_plugin.hpp"
#include "template_executable_network.hpp"
#include "template_infer_request.hpp"

using namespace TemplatePlugin;

// ! [plugin:ctor]
Plugin::Plugin() {
    // TODO: fill with actual device name
    _pluginName = "TEMPLATE";
}
// ! [plugin:ctor]

// ! [plugin:load_exe_network_impl]
InferenceEngine::ExecutableNetworkInternal::Ptr Plugin::LoadExeNetworkImpl(const InferenceEngine::ICNNNetwork & network,
                                                                           const ConfigMap &config) {
    auto cfg = Configuration{ config, _cfg };
    InferenceEngine::InputsDataMap networkInputs;
    InferenceEngine::OutputsDataMap networkOutputs;

    network.getInputsInfo(networkInputs);
    network.getOutputsInfo(networkOutputs);

    // TODO: check with precisions supported by Template device

    for (auto networkOutput : networkOutputs) {
        auto output_precision = networkOutput.second->getPrecision();

        if (output_precision != Precision::FP32 &&
            output_precision != Precision::FP16) {
            THROW_IE_EXCEPTION << "Template device supports only FP16 and FP32 output precision.";
        }
    }

    for (auto networkInput : networkInputs) {
        auto input_precision = networkInput.second->getTensorDesc().getPrecision();

        if (input_precision != InferenceEngine::Precision::FP32 &&
            input_precision != InferenceEngine::Precision::FP16 &&
            input_precision != InferenceEngine::Precision::I16 &&
            input_precision != InferenceEngine::Precision::U8) {
            THROW_IE_EXCEPTION << "Input image format " << input_precision << " is not supported yet.\n"
                       << "Supported formats are: FP32, FP16, I16 and U8.";
        }
    }

    auto clonedNetwork = cloneNet(network);

    return std::make_shared<ExecutableNetwork>(*clonedNetwork, cfg);
}
// ! [plugin:load_exe_network_impl]

// ! [plugin:import_network_impl]
InferenceEngine::ExecutableNetwork Plugin::ImportNetworkImpl(std::istream& model, const std::map<std::string, std::string>& config) {
    // TODO: Import network from stream is not mandatory functionality;
    // Can just throw an exception and remove the code below
    Configuration exportedCfg;

    // some code below which reads exportedCfg from `model` stream
    // ..

    auto cfg = Configuration(config, exportedCfg);

    IExecutableNetwork::Ptr executableNetwork;
    auto exec_network_impl = std::make_shared<ExecutableNetwork>(model, cfg);
    executableNetwork.reset(new ExecutableNetworkBase<ExecutableNetworkInternal>(exec_network_impl),
                            [](InferenceEngine::details::IRelease *p) {p->Release(); });

    return InferenceEngine::ExecutableNetwork{ executableNetwork };
}
// ! [plugin:import_network_impl]

// ! [plugin:query_network]
void Plugin::QueryNetwork(const ICNNNetwork &network, const ConfigMap& config, QueryNetworkResult &res) const {
    Configuration cfg{config, _cfg, false};
    res.rc = StatusCode::OK;

    if (std::shared_ptr<const ngraph::Function> ngraphFunction = network.getFunction()) {
        auto ops = ngraphFunction->get_ordered_ops();
        for (auto&& op : ops) {
            // TODO: investigate if an op is actually supported by Template device
            bool supported = true;
            if (supported) {
                res.supportedLayersMap.insert({ op->get_friendly_name(), GetName() });
            }
        }
    } else {
        THROW_IE_EXCEPTION << "TEMPLATE plugin can query only IR v10 networks";
    }
}
// ! [plugin:query_network]

// ! [plugin:add_extension]
void Plugin::AddExtension(InferenceEngine::IExtensionPtr /*extension*/) {
    // TODO: add extensions if plugin supports extensions
}
// ! [plugin:add_extension]

// ! [plugin:set_config]
void Plugin::SetConfig(const ConfigMap &config) {
    _cfg = Configuration{config, _cfg};
}
// ! [plugin:set_config]

// ! [plugin:get_config]
InferenceEngine::Parameter Plugin::GetConfig(const std::string& name, const std::map<std::string, InferenceEngine::Parameter> & /*options*/) const {
    return _cfg.Get(name);
}
// ! [plugin:get_config]

// ! [plugin:get_metric]
InferenceEngine::Parameter Plugin::GetMetric(const std::string& name, const std::map<std::string, InferenceEngine::Parameter> & options) const {
    if (METRIC_KEY(SUPPORTED_METRICS) == name) {
        std::vector<std::string> supportedMetrics = {
            METRIC_KEY(AVAILABLE_DEVICES),
            METRIC_KEY(SUPPORTED_METRICS),
            METRIC_KEY(SUPPORTED_CONFIG_KEYS),
            METRIC_KEY(FULL_DEVICE_NAME),
            METRIC_KEY(OPTIMIZATION_CAPABILITIES),
            METRIC_KEY(RANGE_FOR_ASYNC_INFER_REQUESTS) };
        IE_SET_METRIC_RETURN(SUPPORTED_METRICS, supportedMetrics);
    } else if (METRIC_KEY(SUPPORTED_CONFIG_KEYS) == name) {
        std::vector<std::string> confiKeys = {
            CONFIG_KEY(DEVICE_ID),
            CONFIG_KEY(PERF_COUNT) };
        IE_SET_METRIC_RETURN(SUPPORTED_CONFIG_KEYS, confiKeys);
    } else if (METRIC_KEY(AVAILABLE_DEVICES) == name) {
        // TODO: fill list of available devices
        std::vector<std::string> availableDevices = { "" };
        IE_SET_METRIC_RETURN(AVAILABLE_DEVICES, availableDevices);
    } else if (METRIC_KEY(FULL_DEVICE_NAME) == name) {
        std::string name = "Template Device Full Name";
        IE_SET_METRIC_RETURN(FULL_DEVICE_NAME, name);
    } else if (METRIC_KEY(OPTIMIZATION_CAPABILITIES) == name) {
        // TODO: fill actual list of supported capabilities: e.g. Template device supports only FP32
        std::vector<std::string> capabilities = { METRIC_VALUE(FP32), TEMPLATE_METRIC_VALUE(HARDWARE_CONVOLUTION) };
        IE_SET_METRIC_RETURN(OPTIMIZATION_CAPABILITIES, capabilities);
    } else if (METRIC_KEY(RANGE_FOR_ASYNC_INFER_REQUESTS) == name) {
        // TODO: fill with actual values
        using uint = unsigned int;
        IE_SET_METRIC_RETURN(RANGE_FOR_ASYNC_INFER_REQUESTS, std::make_tuple(uint{1}, uint{1}, uint{1}));
    } else  {
        THROW_IE_EXCEPTION << "Unsupported device metric: " << name;
    }
}
// ! [plugin:get_metric]

// ! [plugin:create_plugin_engine]
INFERENCE_PLUGIN_API(StatusCode) CreatePluginEngine(IInferencePlugin *&plugin, ResponseDesc *resp) noexcept {
    try {
        plugin = make_ie_compatible_plugin({2, 1, CI_BUILD_NUMBER, "templatePlugin"},
                                           std::make_shared<Plugin>());
        return OK;
    }
    catch (std::exception &ex) {
        return DescriptionBuffer(GENERAL_ERROR, resp) << ex.what();
    }
}
// ! [plugin:create_plugin_engine]
