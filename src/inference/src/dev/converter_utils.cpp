// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "converter_utils.hpp"

#include <ie_blob.h>
#include <ie_common.h>
#include <ie_compound_blob.h>
#include <ie_layouts.h>

#include <fstream>
#include <ie_input_info.hpp>
#include <ie_plugin_config.hpp>
#include <ie_version.hpp>
#include <memory>
#include <openvino/core/except.hpp>
#include <openvino/op/parameter.hpp>
#include <openvino/runtime/exception.hpp>
#include <openvino/runtime/remote_context.hpp>
#include <openvino/runtime/tensor.hpp>

#include "any_copy.hpp"
#include "cnn_network_ngraph_impl.hpp"
#include "cpp_interfaces/interface/ie_iexecutable_network_internal.hpp"
#include "cpp_interfaces/interface/ie_iplugin_internal.hpp"
#include "ie_icore.hpp"
#include "ie_ngraph_utils.hpp"
#include "iplugin_wrapper.hpp"
#include "openvino/runtime/iplugin.hpp"
#include "so_ptr.hpp"
#include "transformations/utils/utils.hpp"

namespace {

void fill_input_info(ov::Output<ov::Node>& input, InferenceEngine::InputInfo::Ptr& input_info) {
    const ov::Output<const ov::Node> const_input(input.get_node(), input.get_index());
    ov::legacy_convert::fill_input_info(const_input, input_info);
    auto& rt_info = input.get_rt_info();
    auto it = rt_info.find("ie_legacy_preproc");
    if (it != rt_info.end()) {
        rt_info.erase(it);
    }
    it = rt_info.find("ie_legacy_td");
    if (it != rt_info.end()) {
        rt_info.erase(it);
    }
}

void fill_output_info(ov::Output<ov::Node>& input, InferenceEngine::DataPtr& output_info) {
    const ov::Output<const ov::Node> const_input(input.get_node(), input.get_index());
    ov::legacy_convert::fill_output_info(const_input, output_info);
    auto& rt_info = input.get_rt_info();
    auto it = rt_info.find("ie_legacy_td");
    if (it != rt_info.end()) {
        rt_info.erase(it);
    }
}

InferenceEngine::SizeVector get_dims(const ov::Output<const ov::Node>& port) {
    InferenceEngine::SizeVector dims = {};
    const auto& p_shape = port.get_partial_shape();
    if (p_shape.is_static())
        dims = p_shape.get_shape();
    return dims;
}

}  // namespace

void ov::legacy_convert::fill_input_info(const ov::Output<const ov::Node>& input,
                                         InferenceEngine::InputInfo::Ptr& input_info) {
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
void ov::legacy_convert::fill_output_info(const ov::Output<const ov::Node>& output,
                                          InferenceEngine::DataPtr& output_info) {
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

InferenceEngine::CNNNetwork ov::legacy_convert::convert_model(const std::shared_ptr<const ov::Model>& model,
                                                              bool is_new_api) {
    auto network = InferenceEngine::CNNNetwork(std::shared_ptr<InferenceEngine::ICNNNetwork>(
        new InferenceEngine::details::CNNNetworkNGraphImpl(model->clone(), {}, is_new_api)));
    std::shared_ptr<ov::Model> cloned_model = network.getFunction();
    for (auto&& input : cloned_model->inputs()) {
        auto param_name = input.get_node()->get_friendly_name();

        OPENVINO_ASSERT(network.getInputsInfo().count(param_name));

        auto input_info = network.getInputsInfo()[param_name];
        ::fill_input_info(input, input_info);
    }
    for (auto&& result : cloned_model->get_results()) {
        auto output = result->input_value(0);
        const auto& res_name = ov::op::util::create_ie_output_name(output);

        OPENVINO_ASSERT(network.getOutputsInfo().count(res_name));
        auto output_info = network.getOutputsInfo()[res_name];

        ::fill_output_info(output, output_info);
    }
    return network;
}
std::shared_ptr<const ov::Model> ov::legacy_convert::convert_model(const InferenceEngine::CNNNetwork& network,
                                                                   bool is_new_api) {
    OPENVINO_ASSERT(network.getFunction(),
                    "CNNNetwork can be converted to OpenVINO Model only in case if it contains ngraph::Function");
    if (is_new_api)
        return network.getFunction();

    auto cloned_model = network.getFunction()->clone();
    for (auto&& input : cloned_model->inputs()) {
        auto param_name = input.get_node()->get_friendly_name();

        OPENVINO_ASSERT(network.getInputsInfo().count(param_name));

        auto input_info = network.getInputsInfo().at(param_name);
        auto& rt_info = input.get_rt_info();
        rt_info["ie_legacy_preproc"] = input_info->getPreProcess();
        rt_info["ie_legacy_td"] = input_info->getTensorDesc();
    }
    for (auto&& result : cloned_model->get_results()) {
        auto output = result->input_value(0);
        const auto& res_name = ov::op::util::create_ie_output_name(output);

        OPENVINO_ASSERT(network.getOutputsInfo().count(res_name));
        auto output_info = network.getOutputsInfo().at(res_name);

        auto& rt_info = output.get_rt_info();
        rt_info["ie_legacy_td"] = output_info->getTensorDesc();
    }
    return cloned_model;
}

namespace ov {

class IInferencePluginWrapper : public InferenceEngine::IInferencePlugin {
public:
    IInferencePluginWrapper(const std::shared_ptr<ov::IPlugin>& plugin) {
        auto& ver = plugin->get_version();
        InferenceEngine::Version version;
        version.buildNumber = ver.buildNumber;
        version.description = ver.description;
        SetVersion(version);
        _isNewAPI = plugin->is_new_api();
        _executorManager = plugin->get_executor_manager();
    }
    std::string GetName() const noexcept override {
        return m_plugin->get_device_name();
    }

    void SetName(const std::string& name) noexcept override {
        m_plugin->set_device_name(name);
    }

    std::shared_ptr<InferenceEngine::IExecutableNetworkInternal> LoadNetwork(
        const InferenceEngine::CNNNetwork& network,
        const std::map<std::string, std::string>& config) override {
        return ov::legacy_convert::convert_compiled_model(
            m_plugin->compile_model(ov::legacy_convert::convert_model(network, m_plugin->is_new_api()),
                                    ov::any_copy(config)));
    }

    std::shared_ptr<InferenceEngine::IExecutableNetworkInternal> LoadNetwork(
        const InferenceEngine::CNNNetwork& network,
        const std::map<std::string, std::string>& config,
        const std::shared_ptr<InferenceEngine::RemoteContext>& context) override {
        return ov::legacy_convert::convert_compiled_model(
            m_plugin->compile_model(ov::legacy_convert::convert_model(network, m_plugin->is_new_api()),
                                    ov::any_copy(config),
                                    ov::RemoteContext{context, {}}));
    }

    ov::SoPtr<InferenceEngine::IExecutableNetworkInternal> LoadNetwork(
        const std::string& modelPath,
        const std::map<std::string, std::string>& config) override {
        return ov::SoPtr<InferenceEngine::IExecutableNetworkInternal>(
            ov::legacy_convert::convert_compiled_model(m_plugin->compile_model(modelPath, ov::any_copy(config))),
            {});
    }

    void AddExtension(const std::shared_ptr<InferenceEngine::IExtension>& extension) override {
        m_plugin->add_extension(extension);
    }

    void SetConfig(const std::map<std::string, std::string>& config) override {
        m_plugin->set_property(ov::any_copy(config));
    }

    void SetProperties(const ov::AnyMap& config) override {
        m_plugin->set_property(config);
    }

    InferenceEngine::Parameter GetConfig(
        const std::string& name,
        const std::map<std::string, InferenceEngine::Parameter>& options) const override {
        return m_plugin->get_property(name, options);
    }

    InferenceEngine::Parameter GetMetric(
        const std::string& name,
        const std::map<std::string, InferenceEngine::Parameter>& options) const override {
        return m_plugin->get_property(name, options);
    }

    std::shared_ptr<InferenceEngine::RemoteContext> CreateContext(const InferenceEngine::ParamMap& params) override {
        return m_plugin->create_context(params)._impl;
    }

    std::shared_ptr<InferenceEngine::RemoteContext> GetDefaultContext(
        const InferenceEngine::ParamMap& params) override {
        return m_plugin->get_default_context(params)._impl;
    }

    std::shared_ptr<InferenceEngine::IExecutableNetworkInternal> ImportNetwork(
        const std::string& modelFileName,
        const std::map<std::string, std::string>& config) override {
        std::ifstream model(modelFileName, std::ios::binary);
        return ov::legacy_convert::convert_compiled_model(m_plugin->import_model(model, ov::any_copy(config)));
    }

    std::shared_ptr<InferenceEngine::IExecutableNetworkInternal> ImportNetwork(
        std::istream& networkModel,
        const std::map<std::string, std::string>& config) override {
        return ov::legacy_convert::convert_compiled_model(m_plugin->import_model(networkModel, ov::any_copy(config)));
    }

    std::shared_ptr<InferenceEngine::IExecutableNetworkInternal> ImportNetwork(
        std::istream& networkModel,
        const std::shared_ptr<InferenceEngine::RemoteContext>& context,
        const std::map<std::string, std::string>& config) override {
        return ov::legacy_convert::convert_compiled_model(
            m_plugin->import_model(networkModel, ov::RemoteContext{context, {}}, ov::any_copy(config)));
    }

    void SetCore(std::weak_ptr<InferenceEngine::ICore> core) override {
        return m_plugin->set_core(std::dynamic_pointer_cast<ov::ICore>(core));
    }

    std::shared_ptr<InferenceEngine::ICore> GetCore() const noexcept override {
        auto core = m_plugin->get_core();
        return std::dynamic_pointer_cast<InferenceEngine::ICore>(core);
    }

    InferenceEngine::QueryNetworkResult QueryNetwork(const InferenceEngine::CNNNetwork& network,
                                                     const std::map<std::string, std::string>& config) const override {
        auto res = m_plugin->query_model(ov::legacy_convert::convert_model(network, m_plugin->is_new_api()),
                                         ov::any_copy(config));
        ie::QueryNetworkResult ret;
        if (!network.getFunction() || res.empty()) {
            ret.rc = InferenceEngine::GENERAL_ERROR;
            return ret;
        }
        ret.supportedLayersMap = res;

        return ret;
    }

    std::shared_ptr<ov::IPlugin> get_plugin() {
        return m_plugin;
    }

private:
    std::shared_ptr<ov::IPlugin> m_plugin;
};

}  // namespace ov

std::shared_ptr<::InferenceEngine::IInferencePlugin> ov::legacy_convert::convert_plugin(
    const std::shared_ptr<::ov::IPlugin>& plugin) {
    if (auto wrapper = std::dynamic_pointer_cast<InferenceEngine::IPluginWrapper>(plugin))
        return wrapper->get_plugin();
    return std::make_shared<ov::IInferencePluginWrapper>(plugin);
}

std::shared_ptr<::ov::IPlugin> ov::legacy_convert::convert_plugin(
    const std::shared_ptr<::InferenceEngine::IInferencePlugin>& plugin) {
    std::shared_ptr<::ov::IPlugin> ov_plugin(new ::InferenceEngine::IPluginWrapper(plugin));
    return ov_plugin;
}

namespace InferenceEngine {

class ICompiledModelWrapper : public ov::ICompiledModel {
public:
    ICompiledModelWrapper(const std::shared_ptr<InferenceEngine::IExecutableNetworkInternal>& model)
        : ov::ICompiledModel(nullptr, ov::legacy_convert::convert_plugin(model->_plugin)),
          m_model(model) {
        std::vector<ov::Output<const ov::Node>> inputs, outputs;
        for (const auto& input : m_model->getInputs()) {
            inputs.emplace_back(input->output(0));
        }
        for (const auto& output : m_model->getOutputs()) {
            outputs.emplace_back(output->output(0));
        }
        m_inputs = inputs;
        m_outputs = outputs;
    }
    std::shared_ptr<InferenceEngine::IInferRequestInternal> create_infer_request() const override {
        return m_model->CreateInferRequest();
    }

    void export_model(std::ostream& model) const override {
        m_model->Export(model);
    }

    std::shared_ptr<ov::Model> get_runtime_model() const override {
        return m_model->GetExecGraphInfo();
    }

    void set_property(const ov::AnyMap& properties) override {
        m_model->SetConfig(properties);
    }

    ov::Any get_property(const std::string& name) const override {
        if (ov::loaded_from_cache == name) {
            return m_model->isLoadedFromCache();
        }
        if (ov::supported_properties == name) {
            try {
                auto supported_properties = m_model->GetMetric(name).as<std::vector<ov::PropertyName>>();
                supported_properties.erase(std::remove_if(supported_properties.begin(),
                                                          supported_properties.end(),
                                                          [](const ov::PropertyName& name) {
                                                              return name == METRIC_KEY(SUPPORTED_METRICS) ||
                                                                     name == METRIC_KEY(SUPPORTED_CONFIG_KEYS);
                                                          }),
                                           supported_properties.end());
                return supported_properties;
            } catch (InferenceEngine::Exception&) {
                auto ro_properties = m_model->GetMetric(METRIC_KEY(SUPPORTED_METRICS)).as<std::vector<std::string>>();
                auto rw_properties =
                    m_model->GetMetric(METRIC_KEY(SUPPORTED_CONFIG_KEYS)).as<std::vector<std::string>>();
                std::vector<ov::PropertyName> supported_properties;
                for (auto&& ro_property : ro_properties) {
                    if (ro_property != METRIC_KEY(SUPPORTED_METRICS) &&
                        ro_property != METRIC_KEY(SUPPORTED_CONFIG_KEYS)) {
                        supported_properties.emplace_back(ro_property, ov::PropertyMutability::RO);
                    }
                }
                for (auto&& rw_property : rw_properties) {
                    supported_properties.emplace_back(rw_property, ov::PropertyMutability::RW);
                }
                supported_properties.emplace_back(ov::supported_properties.name(), ov::PropertyMutability::RO);
                supported_properties.emplace_back(ov::loaded_from_cache.name(), ov::PropertyMutability::RO);
                return supported_properties;
            }
        }
        try {
            return m_model->GetMetric(name);
        } catch (InferenceEngine::Exception&) {
            return m_model->GetConfig(name);
        }
    }

    ov::RemoteContext get_context() const override {
        return {m_model->GetContext(), {}};
    }

    std::shared_ptr<InferenceEngine::IExecutableNetworkInternal> get_model() {
        return m_model;
    }

private:
    std::shared_ptr<InferenceEngine::IExecutableNetworkInternal> m_model;

    std::shared_ptr<InferenceEngine::IInferRequestInternal> create_sync_infer_request() const override {
        OPENVINO_NOT_IMPLEMENTED;
    }
};
}  // namespace InferenceEngine

namespace ov {

class IExecutableNetworkWrapper : public InferenceEngine::IExecutableNetworkInternal {
public:
    IExecutableNetworkWrapper(const std::shared_ptr<ov::ICompiledModel>& model) : m_model(model) {
        for (const auto& input : m_model->inputs()) {
            InferenceEngine::InputInfo::Ptr input_info;
            ov::legacy_convert::fill_input_info(input, input_info);
            _networkInputs[input_info->name()] = input_info;
            _parameters.emplace_back(input.get_node_shared_ptr());
        }
        for (const auto& output : m_model->outputs()) {
            InferenceEngine::DataPtr output_info;
            ov::legacy_convert::fill_output_info(output, output_info);
            _networkOutputs[output_info->getName()] = output_info;
            _results.emplace_back(output.get_node_shared_ptr());
        }
        _plugin = ov::legacy_convert::convert_plugin(std::const_pointer_cast<ov::IPlugin>(m_model->m_plugin));
    }

    std::shared_ptr<InferenceEngine::IInferRequestInternal> CreateInferRequest() override {
        return m_model->create_infer_request();
    }

    void Export(std::ostream& model) override {
        m_model->export_model(model);
    }

    void Export(const std::string& modelFileName) override {
        std::ofstream ostream(modelFileName, std::ios::out | std::ios::binary);
        Export(ostream);
    }

    std::shared_ptr<ngraph::Function> GetExecGraphInfo() override {
        return m_model->get_runtime_model();
    }

    void SetConfig(const std::map<std::string, InferenceEngine::Parameter>& config) override {
        m_model->set_property(config);
    }

    InferenceEngine::Parameter GetConfig(const std::string& name) const override {
        return m_model->get_property(name);
    }

    InferenceEngine::Parameter GetMetric(const std::string& name) const override {
        return m_model->get_property(name);
    }

    std::shared_ptr<InferenceEngine::RemoteContext> GetContext() const override {
        return m_model->get_context()._impl;
    }

    std::shared_ptr<ov::ICompiledModel> get_model() {
        return m_model;
    }

private:
    std::shared_ptr<ov::ICompiledModel> m_model;
};
}  // namespace ov

std::shared_ptr<InferenceEngine::IExecutableNetworkInternal> ov::legacy_convert::convert_compiled_model(
    const std::shared_ptr<ov::ICompiledModel>& model) {
    if (auto comp_model = std::dynamic_pointer_cast<InferenceEngine::ICompiledModelWrapper>(model)) {
        return comp_model->get_model();
    }
    return std::make_shared<ov::IExecutableNetworkWrapper>(model);
}

std::shared_ptr<ov::ICompiledModel> ov::legacy_convert::convert_compiled_model(
    const std::shared_ptr<InferenceEngine::IExecutableNetworkInternal>& model) {
    if (auto comp_model = std::dynamic_pointer_cast<ov::IExecutableNetworkWrapper>(model)) {
        return comp_model->get_model();
    }
    return std::make_shared<InferenceEngine::ICompiledModelWrapper>(model);
}
