// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/iplugin.hpp"

#include <memory>

#include "cpp_interfaces/interface/ie_iexecutable_network_internal.hpp"
#include "cpp_interfaces/interface/ie_iplugin_internal.hpp"
#include "ie_icore.hpp"
#include "openvino/core/any.hpp"
#include "openvino/core/except.hpp"
#include "openvino/core/model.hpp"
#include "openvino/icompiled_model.hpp"
#include "threading/ie_executor_manager.hpp"

ov::IPlugin::IPlugin() : m_executor_manager(InferenceEngine::executorManager()), m_is_new_api(true) {}

ov::IPlugin::IPlugin(const std::shared_ptr<InferenceEngine::IInferencePlugin>& ptr) : old_plugin(ptr) {
    auto& ver = old_plugin->GetVersion();
    m_version.buildNumber = ver.buildNumber;
    m_version.description = ver.description;
    m_plugin_name = old_plugin->GetName();
}

void ov::IPlugin::set_version(const ov::Version& version) {
    if (old_plugin) {
        ie::Version ver;
        ver.buildNumber = version.buildNumber;
        ver.description = version.description;
        old_plugin->SetVersion(ver);
    }
    m_version = version;
}

const ov::Version& ov::IPlugin::get_version() const {
    return m_version;
}

void ov::IPlugin::set_name(const std::string& name) {
    if (old_plugin)
        old_plugin->SetName(name);
    m_plugin_name = name;
}

const std::string& ov::IPlugin::get_name() const {
    return m_plugin_name;
}

std::shared_ptr<InferenceEngine::IExecutableNetworkInternal> ov::IPlugin::compile_model(
    const std::shared_ptr<const ov::Model>& model,
    const ov::AnyMap& properties) {
    return compile_model(model, properties, {});
}

std::shared_ptr<InferenceEngine::IExecutableNetworkInternal> ov::IPlugin::compile_model(
    const std::shared_ptr<const ov::Model>& model,
    const ov::AnyMap& properties,
    const ov::RemoteContext& context) {
    std::shared_ptr<InferenceEngine::IExecutableNetworkInternal> compiled_model;
    if (old_plugin) {
        // TODO: Implement fallback to old behaviour
        return compiled_model;
    }
    std::shared_ptr<ov::Model> cloned_model = model->clone();
    // Instead of:
    // function = std::make_shared<ov::Model>(orig_function->get_results(),
    //                                        orig_function->get_sinks(),
    //                                        orig_function->get_parameters(),
    //                                        orig_function->get_variables(),
    //                                        orig_function->get_friendly_name());
    // function->get_rt_info() = orig_function->get_rt_info();

    if (!is_new_api()) {
        // if IR `version` is not set, suppose it's IR v10 for old API
        // it allows to use operation names in set_ / get_tensor instead of tensor_names
        if (!cloned_model->has_rt_info("version")) {
            cloned_model->set_rt_info(int64_t(10), "version");

            // re-create `network` with new patched `function`
            // TODO: Implement this logic
            // using namespace InferenceEngine;
            // OPENVINO_SUPPRESS_DEPRECATED_START
            // const auto& orig_icnn = static_cast<const ICNNNetwork&>(orig_network);
            // auto orig_impl =
            //     std::dynamic_pointer_cast<const details::CNNNetworkNGraphImpl>(orig_icnn.shared_from_this());
            // OPENVINO_ASSERT(orig_impl != nullptr,
            //                 "Internal: orig_impl must be castable to details::CNNNetworkNGraphImpl");
            // auto new_impl =
            //     std::make_shared<details::CNNNetworkNGraphImpl>(function, orig_impl->getExtensions(), IsNewAPI());
            // network = CNNNetwork(new_impl);
            // for (const auto& inputInfo : orig_network.getInputsInfo()) {
            //     auto toInfo = network.getInputsInfo().at(inputInfo.first);
            //     toInfo->setPrecision(inputInfo.second->getPrecision());
            //     toInfo->setLayout(inputInfo.second->getLayout());
            //     toInfo->getPreProcess() = inputInfo.second->getPreProcess();
            // }
            // for (const auto& outputInfo : orig_network.getOutputsInfo()) {
            //     auto toInfo = network.getOutputsInfo().at(outputInfo.first);
            //     toInfo->setPrecision(outputInfo.second->getPrecision());
            //     toInfo->setLayout(outputInfo.second->getLayout());
            // }
            // OPENVINO_SUPPRESS_DEPRECATED_END
        }
    }

    // TODO:
    // if (context == ov::RemoteContext()) {
    compiled_model = compile_model_impl(cloned_model, properties);
    // } else {
    //     compiled_model = compile_model_impl(cloned_model, context, properties);
    // }
    return compiled_model;
}

std::shared_ptr<InferenceEngine::IExecutableNetworkInternal> ov::IPlugin::compile_model_impl(
    const std::shared_ptr<ov::Model>& model,
    const ov::RemoteContext& context,
    const ov::AnyMap& properties) {
    OPENVINO_NOT_IMPLEMENTED;
}

std::shared_ptr<InferenceEngine::IExecutableNetworkInternal> ov::IPlugin::compile_model_impl(
    const std::shared_ptr<ov::Model>& model,
    const ov::AnyMap& properties) {
    OPENVINO_NOT_IMPLEMENTED;
}

void ov::IPlugin::add_extension(const std::shared_ptr<InferenceEngine::IExtension>& extension) {
    OPENVINO_NOT_IMPLEMENTED;
}

void ov::IPlugin::set_property(const ov::AnyMap& properties) {
    OPENVINO_NOT_IMPLEMENTED;
}

ov::Any ov::IPlugin::get_property(const std::string& name, const ov::AnyMap& arguments) const {
    OPENVINO_NOT_IMPLEMENTED;
}

ov::RemoteContext ov::IPlugin::create_context(const ov::AnyMap& remote_properties) {
    OPENVINO_NOT_IMPLEMENTED;
}

ov::RemoteContext ov::IPlugin::get_default_context(const ov::AnyMap& remote_properties) {
    OPENVINO_NOT_IMPLEMENTED;
}

std::shared_ptr<InferenceEngine::IExecutableNetworkInternal> ov::IPlugin::import_model(std::istream& model,
                                                                                       const ov::AnyMap& properties) {
    OPENVINO_NOT_IMPLEMENTED;
}

std::shared_ptr<InferenceEngine::IExecutableNetworkInternal> ov::IPlugin::import_model(std::istream& model,
                                                                                       const ov::RemoteContext& context,
                                                                                       const ov::AnyMap& properties) {
    OPENVINO_NOT_IMPLEMENTED;
}

void ov::IPlugin::set_core(std::weak_ptr<ov::ICore> core) {
    OPENVINO_ASSERT(!core.expired());
    m_core = core;
    auto locked_core = m_core.lock();
    if (locked_core)
        m_is_new_api = locked_core->isNewAPI();
}

std::shared_ptr<ov::ICore> ov::IPlugin::get_core() const {
    return m_core.lock();
}

bool ov::IPlugin::is_new_api() const {
    return m_is_new_api;
}

const std::shared_ptr<InferenceEngine::ExecutorManager>& ov::IPlugin::get_executor_manager() const {
    return m_executor_manager;
}

ov::SupportedOpsMap ov::IPlugin::query_model(const std::shared_ptr<const ov::Model>& model,
                                             const ov::AnyMap& properties) const {
    OPENVINO_NOT_IMPLEMENTED;
}
