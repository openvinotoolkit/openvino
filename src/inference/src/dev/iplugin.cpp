// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/runtime/iplugin.hpp"

#include <cpp/ie_cnn_network.h>
#include <ie_common.h>
#include <ie_layouts.h>

#include <ie_icnn_network.hpp>
#include <ie_precision.hpp>
#include <ie_preprocess.hpp>
#include <memory>
#include <openvino/core/layout.hpp>
#include <openvino/core/partial_shape.hpp>
#include <openvino/core/preprocess/color_format.hpp>
#include <openvino/core/preprocess/pre_post_process.hpp>
#include <openvino/core/preprocess/resize_algorithm.hpp>
#include <sstream>
#include <vector>

#include "any_copy.hpp"
#include "cnn_network_ngraph_impl.hpp"
#include "converter_utils.hpp"
#include "cpp_interfaces/interface/ie_iexecutable_network_internal.hpp"
#include "cpp_interfaces/interface/ie_iplugin_internal.hpp"
#include "ie_icore.hpp"
#include "ie_ngraph_utils.hpp"
#include "openvino/core/any.hpp"
#include "openvino/core/except.hpp"
#include "openvino/core/model.hpp"
#include "openvino/runtime/icompiled_model.hpp"
#include "threading/ie_executor_manager.hpp"
#include "transformations/utils/utils.hpp"

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

void ov::IPlugin::set_device_name(const std::string& name) {
    if (old_plugin)
        old_plugin->SetName(name);
    m_plugin_name = name;
}

const std::string& ov::IPlugin::get_device_name() const {
    return m_plugin_name;
}

std::shared_ptr<ov::ICompiledModel> ov::IPlugin::compile_model(const std::shared_ptr<const ov::Model>& model,
                                                               const ov::AnyMap& properties) const {
    if (old_plugin) {
        auto exec_network =
            old_plugin->LoadNetwork(ov::legacy_convert::convert_model(model, is_new_api()), any_copy(properties));
        auto compiled_model = ov::legacy_convert::convert_compiled_model(exec_network);
        return compiled_model;
    }
    return compile_model(model, properties, {});
}

std::shared_ptr<ov::ICompiledModel> ov::IPlugin::compile_model(const std::shared_ptr<const ov::Model>& model,
                                                               const ov::AnyMap& properties,
                                                               const ov::RemoteContext& context) const {
    std::shared_ptr<ICompiledModel> compiled_model;
    if (old_plugin) {
        auto compiled_model = ov::legacy_convert::convert_compiled_model(
            old_plugin->LoadNetwork(ov::legacy_convert::convert_model(model, is_new_api()),
                                    any_copy(properties),
                                    context._impl));
        return compiled_model;
    }
    OPENVINO_NOT_IMPLEMENTED;
}

void ov::IPlugin::add_extension(const std::shared_ptr<InferenceEngine::IExtension>& extension) {
    if (old_plugin) {
        old_plugin->AddExtension(extension);
        return;
    }
    OPENVINO_NOT_IMPLEMENTED;
}

void ov::IPlugin::set_property(const ov::AnyMap& properties) {
    if (old_plugin) {
        old_plugin->SetProperties(properties);
        return;
    }
    OPENVINO_NOT_IMPLEMENTED;
}

ov::Any ov::IPlugin::get_property(const std::string& name, const ov::AnyMap& arguments) const {
    if (old_plugin) {
        try {
            return old_plugin->GetConfig(name, arguments);
        } catch (...) {
            return old_plugin->GetMetric(name, arguments);
        }
    }
    OPENVINO_NOT_IMPLEMENTED;
}

ov::RemoteContext ov::IPlugin::create_context(const ov::AnyMap& remote_properties) const {
    if (old_plugin) {
        return ov::RemoteContext{old_plugin->CreateContext(remote_properties), {nullptr}};
    }
    OPENVINO_NOT_IMPLEMENTED;
}

ov::RemoteContext ov::IPlugin::get_default_context(const ov::AnyMap& remote_properties) const {
    if (old_plugin) {
        return ov::RemoteContext{old_plugin->GetDefaultContext(remote_properties), {nullptr}};
    }
    OPENVINO_NOT_IMPLEMENTED;
}

std::shared_ptr<ov::ICompiledModel> ov::IPlugin::import_model(std::istream& model, const ov::AnyMap& properties) const {
    if (old_plugin) {
        return ov::legacy_convert::convert_compiled_model(old_plugin->ImportNetwork(model, any_copy(properties)));
    }
    OPENVINO_NOT_IMPLEMENTED;
}

std::shared_ptr<ov::ICompiledModel> ov::IPlugin::import_model(std::istream& model,
                                                              const ov::RemoteContext& context,
                                                              const ov::AnyMap& properties) const {
    if (old_plugin) {
        return ov::legacy_convert::convert_compiled_model(
            old_plugin->ImportNetwork(model, context._impl, any_copy(properties)));
    }
    OPENVINO_NOT_IMPLEMENTED;
}

void ov::IPlugin::set_core(std::weak_ptr<ov::ICore> core) {
    OPENVINO_ASSERT(!core.expired());
    m_core = core;
    auto locked_core = m_core.lock();
    if (old_plugin) {
        auto old_core = std::dynamic_pointer_cast<InferenceEngine::ICore>(locked_core);
        if (old_core)
            old_plugin->SetCore(old_core);
    }
    if (locked_core)
        m_is_new_api = locked_core->is_new_api();
}

std::shared_ptr<ov::ICore> ov::IPlugin::get_core() const {
    if (old_plugin)
        return old_plugin->GetCore();
    return m_core.lock();
}

bool ov::IPlugin::is_new_api() const {
    if (old_plugin)
        return old_plugin->IsNewAPI();
    return m_is_new_api;
}

const std::shared_ptr<InferenceEngine::ExecutorManager>& ov::IPlugin::get_executor_manager() const {
    if (old_plugin)
        return old_plugin->executorManager();
    return m_executor_manager;
}

ov::SupportedOpsMap ov::IPlugin::query_model(const std::shared_ptr<const ov::Model>& model,
                                             const ov::AnyMap& properties) const {
    if (old_plugin) {
        auto res =
            old_plugin->QueryNetwork(ov::legacy_convert::convert_model(model, is_new_api()), any_copy(properties));
        if (res.rc != InferenceEngine::OK) {
            throw ov::Exception(res.resp.msg);
        }
        return res.supportedLayersMap;
    }
    OPENVINO_NOT_IMPLEMENTED;
}
