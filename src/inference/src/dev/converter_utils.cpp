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
#include <ie_version.hpp>
#include <memory>
#include <openvino/core/except.hpp>
#include <openvino/op/parameter.hpp>
#include <openvino/runtime/exception.hpp>
#include <openvino/runtime/remote_context.hpp>
#include <openvino/runtime/tensor.hpp>

#include "any_copy.hpp"
#include "cnn_network_ngraph_impl.hpp"
#include "cpp/ie_plugin.hpp"
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

InferenceEngine::SizeVector get_dims(const ov::Output<const ov::Node>& port,
                                     const std::function<bool(InferenceEngine::SizeVector& dims)>& callback = {}) {
    InferenceEngine::SizeVector dims = {};
    const auto& p_shape = port.get_partial_shape();
    if (p_shape.is_static())
        dims = p_shape.get_shape();
    else {
        if (!callback || !callback(dims)) {
            if (p_shape.rank().is_static()) {
                for (size_t i = 0; i < static_cast<size_t>(p_shape.rank().get_length()); i++) {
                    dims.emplace_back(0);
                }
            }
        }
    }
    return dims;
}

}  // namespace

void ov::legacy_convert::fill_input_info(const ov::Output<const ov::Node>& input,
                                         InferenceEngine::InputInfo::Ptr& input_info) {
    if (!input_info) {
        // Create input info
        auto param_name = input.get_node()->get_friendly_name();
        auto dims = get_dims(input, [&](InferenceEngine::SizeVector& dims) -> bool {
            auto param = std::dynamic_pointer_cast<const ov::op::v0::Parameter>(input.get_node_shared_ptr());
            if (param && param->get_partial_shape().is_static()) {
                dims = param->get_partial_shape().get_shape();
                return true;
            }
            return false;
        });
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

        OPENVINO_ASSERT(network.getInputsInfo().find(param_name) != network.getInputsInfo().end());

        auto input_info = network.getInputsInfo()[param_name];
        ::fill_input_info(input, input_info);
    }
    for (auto&& result : cloned_model->get_results()) {
        auto output = result->input_value(0);
        const auto& res_name = ov::op::util::create_ie_output_name(output);

        OPENVINO_ASSERT(network.getOutputsInfo().find(res_name) != network.getOutputsInfo().end());
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

        OPENVINO_ASSERT(network.getInputsInfo().find(param_name) != network.getInputsInfo().end());

        auto input_info = network.getInputsInfo().at(param_name);
        auto& rt_info = input.get_rt_info();
        rt_info["ie_legacy_preproc"] = input_info->getPreProcess();
        rt_info["ie_legacy_td"] = input_info->getTensorDesc();
    }
    for (auto&& result : cloned_model->get_results()) {
        auto output = result->input_value(0);
        const auto& res_name = ov::op::util::create_ie_output_name(output);

        OPENVINO_ASSERT(network.getOutputsInfo().find(res_name) != network.getOutputsInfo().end());
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
        return m_plugin->compile_model(ov::legacy_convert::convert_model(network, m_plugin->is_new_api()),
                                       ov::any_copy(config));
    }

    std::shared_ptr<InferenceEngine::IExecutableNetworkInternal> LoadNetwork(
        const InferenceEngine::CNNNetwork& network,
        const std::map<std::string, std::string>& config,
        const std::shared_ptr<InferenceEngine::RemoteContext>& context) override {
        return m_plugin->compile_model(ov::legacy_convert::convert_model(network, m_plugin->is_new_api()),
                                       ov::any_copy(config),
                                       ov::RemoteContext{context, {}});
    }

    ov::SoPtr<InferenceEngine::IExecutableNetworkInternal> LoadNetwork(
        const std::string& modelPath,
        const std::map<std::string, std::string>& config) override {
        return ov::SoPtr<InferenceEngine::IExecutableNetworkInternal>(
            m_plugin->compile_model(modelPath, ov::any_copy(config)),
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
        return m_plugin->import_model(model, ov::any_copy(config));
    }

    std::shared_ptr<InferenceEngine::IExecutableNetworkInternal> ImportNetwork(
        std::istream& networkModel,
        const std::map<std::string, std::string>& config) override {
        return m_plugin->import_model(networkModel, ov::any_copy(config));
    }

    std::shared_ptr<InferenceEngine::IExecutableNetworkInternal> ImportNetwork(
        std::istream& networkModel,
        const std::shared_ptr<InferenceEngine::RemoteContext>& context,
        const std::map<std::string, std::string>& config) override {
        return m_plugin->import_model(networkModel, ov::RemoteContext{context, {}}, ov::any_copy(config));
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
