// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///////////////////////////////////////////////////////////////////////////////////////////////////
#include "compile_model.hpp"
#include "common.hpp"
#include <memory>

#include "async_infer_request.hpp"
#include "itt.hpp"
#include "openvino/op/util/op_types.hpp"
#include "openvino/runtime/exec_model_info.hpp"
#include "openvino/runtime/properties.hpp"
#include "plugin.hpp"
#include "transformations/rt_info/fused_names_attribute.hpp"
#include "transformations/utils/utils.hpp"

ov::auto_plugin::CompiledModel::CompiledModel(const std::shared_ptr<ov::Model>& model,
                                              const std::shared_ptr<const ov::IPlugin>& plugin,
                                              ScheduleContext::Ptr context,
                                              Schedule::Ptr        scheduler)
    : ov::ICompiledModel(model, plugin),
      m_context(context),
      m_scheduler(scheduler) {
}

ov::auto_plugin::ISyncInferPtr ov::auto_plugin::CompiledModel::create_sync_infer_request() const {
    return m_scheduler->create_sync_infer_request();
}

ov::auto_plugin::IASyncInferPtr ov::auto_plugin::CompiledModel::create_infer_request() const {
    const_cast<CompiledModel*>(this)->set_compilemodel_for_context();
    auto internal_request = create_sync_infer_request();
    auto async_infer_request = std::make_shared<AsyncInferRequest>(
        m_scheduler,
        std::static_pointer_cast<InferRequest>(internal_request),
        get_callback_executor());
    return async_infer_request;
}

std::string ov::auto_plugin::CompiledModel::get_log_tag() const noexcept {
    return m_context->m_log_tag;
}

ov::AnyMap ov::auto_plugin::CompiledModel::get_device_supported_metrics(AutoLoadContext& context) {
     ov::AnyMap all_devices;
    ov::AnyMap device_properties = {};
    OPENVINO_ASSERT(context.m_exe_network);
    auto device_supported_metrics = context.m_exe_network->get_property(METRIC_KEY(SUPPORTED_METRICS));
    for (auto&& property_name : device_supported_metrics.as<std::vector<std::string>>()) {
        device_properties[property_name] = context.m_exe_network->get_property(property_name);
    }
    auto device_supported_configs = context.m_exe_network->get_property(METRIC_KEY(SUPPORTED_CONFIG_KEYS));
    for (auto&& property_name : device_supported_configs.as<std::vector<std::string>>()) {
        device_properties[property_name] = context.m_exe_network->get_property(property_name);
    }
    all_devices[context.m_device_info.device_name] = device_properties;
    return all_devices;
}

void ov::auto_plugin::CompiledModel::set_compilemodel_for_context() {
    std::call_once(m_oc, [this]() {
        m_context->m_compiled_model = shared_from_this();
    });
}

std::shared_ptr<const ov::auto_plugin::Plugin> ov::auto_plugin::CompiledModel::get_auto_plugin() {
    auto plugin = get_plugin();
    OPENVINO_ASSERT(plugin);
    auto auto_plugin = std::static_pointer_cast<const ov::auto_plugin::Plugin>(plugin);
    OPENVINO_ASSERT(auto_plugin);
    return auto_plugin;
}

