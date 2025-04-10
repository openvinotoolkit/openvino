// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "compiled_model.hpp"

#include <memory>

#include "async_infer_request.hpp"
#include "itt.hpp"
#include "openvino/op/util/op_types.hpp"
#include "openvino/runtime/exec_model_info.hpp"
#include "openvino/runtime/properties.hpp"
#include "perf_counter.hpp"
#include "plugin.hpp"
#include "transformations/rt_info/fused_names_attribute.hpp"
#include "transformations/utils/utils.hpp"

// ! [compiled_model:ctor]
ov::template_plugin::CompiledModel::CompiledModel(const std::shared_ptr<ov::Model>& model,
                                                  const std::shared_ptr<const ov::IPlugin>& plugin,
                                                  const ov::SoPtr<ov::IRemoteContext>& context,
                                                  const std::shared_ptr<ov::threading::ITaskExecutor>& task_executor,
                                                  const Configuration& cfg,
                                                  bool loaded_from_cache)
    : ov::ICompiledModel(model, plugin, context, task_executor),  // Disable default threads creation
      m_cfg(cfg),
      m_model(model),
      m_loaded_from_cache(loaded_from_cache) {
    // TODO: if your plugin supports device ID (more that single instance of device can be on host machine)
    // you should select proper device based on KEY_DEVICE_ID or automatic behavior
    // In this case, m_wait_executor should also be created per device.
    try {
        compile_model(m_model);
    } catch (const std::exception& e) {
        OPENVINO_THROW("Standard exception from compilation library: ", e.what());
    } catch (...) {
        OPENVINO_THROW("Generic exception is thrown");
    }
}
// ! [compiled_model:ctor]

// ! [compiled_model:compile_model]
// forward declaration
void transform_model(const std::shared_ptr<ov::Model>& model);

void ov::template_plugin::CompiledModel::compile_model(const std::shared_ptr<ov::Model>& model) {
    // apply plugins transformations
    if (!m_cfg.disable_transformations)
        transform_model(model);

    // Integrate performance counters to the compiled model
    for (const auto& op : model->get_ops()) {
        auto& rt_info = op->get_rt_info();
        rt_info[ov::runtime::interpreter::PERF_COUNTER_NAME] =
            std::make_shared<ov::runtime::interpreter::PerfCounter>();
    }

    // Perform any other steps like allocation and filling backend specific memory handles and so on
}
// ! [compiled_model:compile_model]

// ! [compiled_model:create_sync_infer_request]
std::shared_ptr<ov::ISyncInferRequest> ov::template_plugin::CompiledModel::create_sync_infer_request() const {
    return std::make_shared<InferRequest>(
        std::static_pointer_cast<const ov::template_plugin::CompiledModel>(shared_from_this()));
}
// ! [compiled_model:create_sync_infer_request]

// ! [compiled_model:create_infer_request]
std::shared_ptr<ov::IAsyncInferRequest> ov::template_plugin::CompiledModel::create_infer_request() const {
    auto internal_request = create_sync_infer_request();
    auto async_infer_request = std::make_shared<AsyncInferRequest>(
        std::static_pointer_cast<ov::template_plugin::InferRequest>(internal_request),
        get_task_executor(),
        get_template_plugin()->m_waitExecutor,
        get_callback_executor());

    return async_infer_request;
}
// ! [compiled_model:create_infer_request]

// ! [compiled_model:set_property]
void ov::template_plugin::CompiledModel::set_property(const ov::AnyMap& properties) {
    m_cfg = Configuration{properties, m_cfg};
}
// ! [compiled_model:set_property]

// ! [compiled_model:get_runtime_model]
std::shared_ptr<const ov::Model> ov::template_plugin::CompiledModel::get_runtime_model() const {
    auto model = m_model->clone();
    // Add execution information into the model
    size_t exec_order = 0;
    for (const auto& op : model->get_ordered_ops()) {
        auto& info = op->get_rt_info();
        const auto& it = info.find(ov::runtime::interpreter::PERF_COUNTER_NAME);
        OPENVINO_ASSERT(it != info.end(), "Operation ", op, " doesn't contain performance counter");
        auto perf_count = it->second.as<std::shared_ptr<ov::runtime::interpreter::PerfCounter>>();
        OPENVINO_ASSERT(perf_count, "Performance counter is empty");
        info[ov::exec_model_info::LAYER_TYPE] = op->get_type_info().name;
        info[ov::exec_model_info::EXECUTION_ORDER] = std::to_string(exec_order++);
        info[ov::exec_model_info::IMPL_TYPE] = "ref";
        info[ov::exec_model_info::PERF_COUNTER] = m_cfg.perf_count && perf_count && perf_count->avg() != 0
                                                      ? std::to_string(perf_count->avg())
                                                      : "not_executed";

        std::string original_names = ov::getFusedNames(op);
        if (original_names.empty()) {
            original_names = op->get_friendly_name();
        } else if (original_names.find(op->get_friendly_name()) == std::string::npos) {
            original_names = op->get_friendly_name() + "," + original_names;
        }
        info[ov::exec_model_info::ORIGINAL_NAMES] = original_names;
    }
    return model;
}
// ! [compiled_model:get_runtime_model]

std::shared_ptr<const ov::template_plugin::Plugin> ov::template_plugin::CompiledModel::get_template_plugin() const {
    auto plugin = get_plugin();
    OPENVINO_ASSERT(plugin);
    auto template_plugin = std::static_pointer_cast<const ov::template_plugin::Plugin>(plugin);
    OPENVINO_ASSERT(template_plugin);
    return template_plugin;
}

// ! [compiled_model:get_property]
ov::Any ov::template_plugin::CompiledModel::get_property(const std::string& name) const {
    const auto& default_ro_properties = []() {
        std::vector<ov::PropertyName> ro_properties{ov::model_name,
                                                    ov::supported_properties,
                                                    ov::execution_devices,
                                                    ov::loaded_from_cache,
                                                    ov::optimal_number_of_infer_requests};
        return ro_properties;
    };
    const auto& default_rw_properties = []() {
        std::vector<ov::PropertyName> rw_properties{ov::device::id, ov::enable_profiling, ov::hint::performance_mode};
        return rw_properties;
    };
    if (ov::model_name == name) {
        auto& model_name = m_model->get_friendly_name();
        return decltype(ov::model_name)::value_type(model_name);
    } else if (ov::loaded_from_cache == name) {
        return m_loaded_from_cache;
    } else if (ov::execution_devices == name) {
        return decltype(ov::execution_devices)::value_type{get_plugin()->get_device_name() + "." +
                                                           std::to_string(m_cfg.device_id)};
    } else if (ov::optimal_number_of_infer_requests == name) {
        unsigned int value = m_cfg.streams;
        return decltype(ov::optimal_number_of_infer_requests)::value_type(value);
    } else if (ov::supported_properties == name) {
        auto ro_properties = default_ro_properties();
        auto rw_properties = default_rw_properties();

        auto supported_properties = decltype(ov::supported_properties)::value_type();
        supported_properties.reserve(ro_properties.size() + rw_properties.size());
        supported_properties.insert(supported_properties.end(), ro_properties.begin(), ro_properties.end());
        supported_properties.insert(supported_properties.end(), rw_properties.begin(), rw_properties.end());
        return supported_properties;
    }

    return m_cfg.Get(name);
}
// ! [compiled_model:get_property]

// ! [compiled_model:export_model]
void ov::template_plugin::CompiledModel::export_model(std::ostream& model_stream) const {
    OV_ITT_SCOPED_TASK(itt::domains::TemplatePlugin, "CompiledModel::export_model");

    std::stringstream xmlFile, binFile;
    ov::pass::Serialize serializer(xmlFile, binFile);
    serializer.run_on_model(m_model);

    auto m_constants = binFile.str();
    auto m_model = xmlFile.str();

    auto dataSize = static_cast<std::uint64_t>(m_model.size());
    model_stream.write(reinterpret_cast<char*>(&dataSize), sizeof(dataSize));
    model_stream.write(m_model.c_str(), dataSize);

    if (m_cfg.cache_mode == CacheMode::OPTIMIZE_SPEED) {
        dataSize = static_cast<std::uint64_t>(m_constants.size());
        model_stream.write(reinterpret_cast<char*>(&dataSize), sizeof(dataSize));
        model_stream.write(reinterpret_cast<char*>(&m_constants[0]), dataSize);
    }
}
// ! [compiled_model:export_model]
