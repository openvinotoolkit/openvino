// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///////////////////////////////////////////////////////////////////////////////////////////////////
#include "auto_batch_plugin.hpp"

#include <cmath>
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "auto_batch_compiled_model.hpp"
#include "dimension_tracker.hpp"
#include "openvino/pass/manager.hpp"
#include "openvino/runtime/intel_gpu/properties.hpp"
#include "openvino/util/common_util.hpp"
#include "transformations/common_optimizations/dimension_tracking.hpp"
#include "transformations/init_node_info.hpp"
#include "transformations/utils/utils.hpp"

std::vector<std::string> supported_configKeys = {CONFIG_KEY(AUTO_BATCH_DEVICE_CONFIG),
                                                 ov::device::priorities.name(),
                                                 ov::auto_batch_timeout.name(),
                                                 ov::cache_dir.name()};

inline ov::AnyMap merge_properties(ov::AnyMap config, const ov::AnyMap& user_config) {
    for (auto&& kvp : user_config) {
        config[kvp.first] = kvp.second;
    }
    return config;
}

ov::autobatch_plugin::Plugin::Plugin() {
    set_device_name("BATCH");
    m_plugin_config[ov::auto_batch_timeout.name()] = "1000";  // default value (ms)
}

void ov::autobatch_plugin::Plugin::set_property(const ov::AnyMap& properties) {
    for (auto&& c : properties) {
        const auto& name = c.first;
        const auto& val = c.second;
        if (supported_configKeys.end() == std::find(supported_configKeys.begin(), supported_configKeys.end(), name))
            OPENVINO_THROW("Unsupported config key: ", name);
        if (name == CONFIG_KEY(AUTO_BATCH_DEVICE_CONFIG) || name == ov::device::priorities.name()) {
            parse_batch_device(val.as<std::string>());
        } else if (name == ov::auto_batch_timeout.name()) {
            try {
                auto t = val.as<uint32_t>();
                if (t < 0)
                    OPENVINO_THROW("Parameter Mismatch!!!");
            } catch (const std::exception&) {
                OPENVINO_THROW(" Expecting unsigned int value for ",
                               ov::auto_batch_timeout.name(),
                               " got ",
                               val.as<uint32_t>());
            }
        }
        m_plugin_config[name] = val;
    }
}

ov::Any ov::autobatch_plugin::Plugin::get_property(const std::string& name, const ov::AnyMap& arguments) const {
    if (supported_configKeys.end() != std::find(supported_configKeys.begin(), supported_configKeys.end(), name)) {
        auto it = m_plugin_config.find(name);
        if (it == m_plugin_config.end()) {
            OPENVINO_THROW("The Value is not set for ", name);
        } else {
            return {it->second};
        }
    } else if (name == METRIC_KEY(SUPPORTED_METRICS)) {
        return std::vector<std::string>{METRIC_KEY(SUPPORTED_METRICS),
                                        ov::device::full_name.name(),
                                        METRIC_KEY(SUPPORTED_CONFIG_KEYS)};
    } else if (name == ov::supported_properties.name()) {
        return std::vector<ov::PropertyName>{
            ov::PropertyName{ov::supported_properties.name(), ov::PropertyMutability::RO},
            ov::PropertyName{ov::device::full_name.name(), ov::PropertyMutability::RO}};
    } else if (name == ov::device::full_name.name()) {
        return get_device_name();
    } else if (name == METRIC_KEY(SUPPORTED_CONFIG_KEYS)) {
        return supported_configKeys;
    } else {
        OPENVINO_THROW("Unsupported property: ", name);
    }
}

std::shared_ptr<ov::ICompiledModel> ov::autobatch_plugin::Plugin::compile_model(
    const std::shared_ptr<const ov::Model>& model,
    const ov::AnyMap& properties) const {
    return compile_model(model, properties, {});
}

std::shared_ptr<ov::ICompiledModel> ov::autobatch_plugin::Plugin::compile_model(
    const std::shared_ptr<const ov::Model>& model,
    const ov::AnyMap& properties,
    const ov::RemoteContext& context) const {
    auto core = get_core();
    if (core == nullptr) {
        OPENVINO_THROW("Please, work with Auto-Batching device via InferencEngine::Core object");
    }

    // merge configs from func properties and m_plugin_config
    auto full_properties = merge_properties(m_plugin_config, properties);

    auto device_batch = full_properties.find(CONFIG_KEY(AUTO_BATCH_DEVICE_CONFIG));
    if (device_batch == full_properties.end())
        device_batch = full_properties.find(ov::device::priorities.name());
    if (device_batch == full_properties.end()) {
        OPENVINO_THROW("KEY_AUTO_BATCH key is not set for BATCH device");
    }

    auto meta_device = parse_meta_device(device_batch->second.as<std::string>(), properties);

    const auto& device_name = meta_device.device_name;
    const auto& device_config = meta_device.device_config;
    auto device_config_no_auto_batch = device_config;
    // avoid recursive auto-batching
    device_config_no_auto_batch[ov::hint::allow_auto_batching.name()] = false;

    std::set<std::string> batched_inputs;
    std::set<std::string> batched_outputs;
    // check that the auto-batching is applicable in general
    try {
        // if applicable, the Auto-Batching is implicitly enabled via the performance hints
        const bool enable_tput_plugin =
            core->get_property(device_name, ov::hint::performance_mode) == ov::hint::PerformanceMode::THROUGHPUT;
        const auto& performance_mode = device_config.find(ov::hint::performance_mode.name());
        const bool enable_tput_cfg = (performance_mode != device_config.end() &&
                                      performance_mode->second == ov::hint::PerformanceMode::THROUGHPUT);
        // if the auto-batching is enabled implicitly, check the dims carefully, to avoid outstanding failures
        const bool check_dims = (enable_tput_plugin || enable_tput_cfg);
        // find the batch dim
        auto cloned_model = model->clone();
        ov::pass::Manager pass_manager;
        pass_manager.register_pass<ov::pass::InitNodeInfo>();
        pass_manager.register_pass<ov::pass::FindBatch>(false, check_dims);
        pass_manager.run_passes(cloned_model);
        // do not reshape/re-batch originally batched networks and when there are no inputs with the N* layouts
        // input(s) should have the batch dim as the first dim (current limitation of the auto-batching impl)
        const auto& params = cloned_model->get_parameters();
        for (size_t input_id = 0; input_id < params.size(); input_id++) {
            const auto& input = params[input_id];
            const auto& shape = input->get_partial_shape();
            // currently no plugin support batched execution for dynamic networks
            if (shape.is_dynamic())
                OPENVINO_THROW("Auto-batching does not support dynamic networks!");
            // check the batch dim: either 0th (and the original batch size of 1) or none
            if (shape.size() && ov::DimensionTracker::get_label(shape[0])) {
                const auto& static_shape = input->get_shape();
                if (static_shape[0] != 1)
                    OPENVINO_THROW("Auto-batching does not reshape/re-batch originally batched networks!");
                batched_inputs.insert(
                    ov::op::util::get_ie_output_name(params[input_id]->output(0)));  // batched dim for the input
            } else {
                // if the 0-th dim is not for the batch, then we support only the case when NONE dimension is batch
                for (size_t s = 1; s < shape.size(); s++)
                    if (ov::DimensionTracker::get_label(shape[s]))
                        OPENVINO_THROW(
                            "Auto-batching operates only networks with inputs/outputs batched by 0th dimension");
            }
        }
        const auto& results = cloned_model->get_results();
        for (size_t output_id = 0; output_id < results.size(); output_id++) {
            const auto& output = results[output_id];
            const auto& shape = output->get_output_partial_shape(0);
            if (shape.is_dynamic())
                OPENVINO_THROW("Auto-batching does not support dynamic networks!");
            // check the batch dim: either 0th (and the original batch size of 1) or none
            if (shape.size() && ov::DimensionTracker::get_label(shape[0])) {
                if (shape[0] != 1)
                    OPENVINO_THROW("Auto-batching does not reshape/re-batch originally batched networks!");
                const auto& node = output->input_value(0);
                batched_outputs.insert(
                    ov::op::util::get_ie_output_name(ov::Output<const ov::Node>(node.get_node(), node.get_index())));
            } else {
                // if the 0-th dim is not for the batch, then we support only the case when NONE dimension is batch
                for (size_t s = 1; s < shape.size(); s++)
                    if (ov::DimensionTracker::get_label(shape[s]))
                        OPENVINO_THROW("Auto-batching operates only networks with outputs batched by 0th dimension");
            }
        }
        if (!batched_inputs.size() || !batched_outputs.size())
            OPENVINO_THROW("Auto-batching supports only networks with inputs/outputs featuring batched dim!");
    } catch (const ov::Exception&) {
        meta_device.device_batch_size = 1;
    }

    if (!meta_device.device_batch_size) {
        // batch size is not set explicitly via device name e.g. BATCH:GPU(4)
        // let's query the optimal batch size
        ov::AnyMap options;
        options["MODEL_PTR"] = std::const_pointer_cast<ngraph::Function>(model);
        auto opt_batch_size =
            core->get_property(device_name, ov::optimal_batch_size.name(), options).as<unsigned int>();
        auto requests = core->get_property(device_name, ov::hint::num_requests);
        const auto& reqs = properties.find(ov::hint::num_requests.name());
        if (reqs != properties.end())
            requests = reqs->second.as<unsigned int>();
        if (requests)
            opt_batch_size = std::max(1u, std::min(requests, opt_batch_size));
        if (opt_batch_size >
            2)  // batching is usually in-efficient for batch<4 (as batch1 kernels are heavily optimized)
            meta_device.device_batch_size = opt_batch_size;
        else
            meta_device.device_batch_size = 1;
    }

    auto report_footprint = [](std::shared_ptr<ICore> pCore, std::string device) -> size_t {
        size_t footprint = 0;
        // TODO: use the per-network metric (22.2) rather than plugin-level
        auto stats = pCore->get_property(device, ov::intel_gpu::memory_statistics);
        for (const auto& s : stats)
            footprint += s.second;
        return footprint;
    };

    size_t batch1_footprint = 0;
    if (device_name.find("GPU") != std::string::npos)
        batch1_footprint = report_footprint(core, device_name);
    auto compiledmodel_without_batch = !context.is_empty()
                                           ? core->compile_model(model, context, device_config_no_auto_batch)
                                           : core->compile_model(model, device_name, device_config_no_auto_batch);
    if (device_name.find("GPU") != std::string::npos) {
        batch1_footprint = report_footprint(core, device_name) - batch1_footprint;
        if (batch1_footprint) {
            const auto total_mem = core->get_property(device_name, ov::intel_gpu::device_total_mem_size);
            const int estimated_batch = static_cast<int>((total_mem - batch1_footprint) / batch1_footprint);
            int closest = static_cast<int>(pow(2, floor(std::log(estimated_batch) / std::log(2))));
            closest = std::max(1, closest);
            meta_device.device_batch_size = std::min(static_cast<int>(meta_device.device_batch_size), closest);
        }
    }

    // auto-batch settings
    ov::AnyMap compiledmodel_config;
    for (const auto& c : full_properties) {
        if (supported_configKeys.end() != std::find(supported_configKeys.begin(), supported_configKeys.end(), c.first))
            compiledmodel_config.insert(c);
    }
    ov::SoPtr<ov::ICompiledModel> compiledmodel_with_batch;
    auto reshaped = model->clone();
    if (meta_device.device_batch_size > 1 && batched_inputs.size()) {
        try {
            auto inputs = reshaped->inputs();
            std::map<ov::Output<ov::Node>, ov::PartialShape> partial_shapes;
            for (auto& input : inputs) {
                auto input_shape = input.get_shape();
                if (batched_inputs.find(ov::op::util::get_ie_output_name(input)) != batched_inputs.end()) {
                    input_shape[0] = meta_device.device_batch_size;
                }
                partial_shapes.insert({input, ov::PartialShape(input_shape)});
            }
            reshaped->reshape(partial_shapes);
            {
                auto new_inputs = reshaped->inputs();
                for (auto& input : inputs) {
                    auto new_input_shape = input.get_shape();
                }
            }
            compiledmodel_with_batch = !context.is_empty()
                                           ? core->compile_model(reshaped, context, device_config_no_auto_batch)
                                           : core->compile_model(reshaped, device_name, device_config_no_auto_batch);
        } catch (const ov::Exception&) {
            meta_device.device_batch_size = 1;
        }
    }
    return std::make_shared<ov::autobatch_plugin::CompiledModel>(model->clone(),
                                                                 shared_from_this(),
                                                                 compiledmodel_config,
                                                                 meta_device,
                                                                 batched_inputs,
                                                                 batched_outputs,
                                                                 compiledmodel_with_batch,
                                                                 compiledmodel_without_batch);

    // Todo create auto batch executable network
}

ov::SupportedOpsMap ov::autobatch_plugin::Plugin::query_model(const std::shared_ptr<const ov::Model>& model,
                                                              const ov::AnyMap& properties) const {
    OPENVINO_ASSERT(model, "OpenVINO Model is empty!");
    OPENVINO_ASSERT(get_core(), "Core is missing!");
    auto cfg = properties;
    for (const auto& c : cfg) {
        if (c.first == CONFIG_KEY(AUTO_BATCH_DEVICE_CONFIG) || c.first == ov::device::priorities.name()) {
            auto val = c.second;
            cfg.erase(c.first);
            auto metaDevice = parse_meta_device(val.as<std::string>(), cfg);
            return get_core()->query_model(model, metaDevice.device_name, cfg);
        }
    }
    OPENVINO_THROW("Value for KEY_AUTO_BATCH_DEVICE_CONFIG is not set");
}

std::shared_ptr<ov::IRemoteContext> ov::autobatch_plugin::Plugin::create_context(
    const ov::AnyMap& remote_properties) const {
    OPENVINO_NOT_IMPLEMENTED;
}

std::shared_ptr<ov::IRemoteContext> ov::autobatch_plugin::Plugin::get_default_context(
    const ov::AnyMap& remote_properties) const {
    OPENVINO_NOT_IMPLEMENTED;
}

std::shared_ptr<ov::ICompiledModel> ov::autobatch_plugin::Plugin::import_model(std::istream& model,
                                                                               const ov::AnyMap& properties) const {
    OPENVINO_NOT_IMPLEMENTED;
}

std::shared_ptr<ov::ICompiledModel> ov::autobatch_plugin::Plugin::import_model(std::istream& model,
                                                                               const ov::RemoteContext& context,
                                                                               const ov::AnyMap& properties) const {
    OPENVINO_NOT_IMPLEMENTED;
}

ov::autobatch_plugin::DeviceInformation ov::autobatch_plugin::Plugin::parse_batch_device(
    const std::string& device_with_Batch) {
    auto&& d = device_with_Batch;
    auto openingBracket = d.find_first_of('(');
    auto closingBracket = d.find_first_of(')', openingBracket);
    auto deviceName = d.substr(0, openingBracket);

    int batch = 0;
    if (closingBracket != std::string::npos && openingBracket < closingBracket) {
        batch = std::stol(d.substr(openingBracket + 1, closingBracket - 1));

        if (batch <= 0) {
            OPENVINO_THROW("Batch value for '", deviceName, "' must be > 0, while ", batch, "is passed");
        }
    }
    return {deviceName, {{}}, static_cast<uint32_t>(batch)};
}

ov::autobatch_plugin::DeviceInformation ov::autobatch_plugin::Plugin::parse_meta_device(
    const std::string& devices_batch_config,
    const ov::AnyMap& user_config) const {
    auto meta_device = parse_batch_device(devices_batch_config);
    meta_device.device_config = get_core()->get_supported_property(meta_device.device_name, user_config);
    // check that no irrelevant config-keys left
    for (const auto& k : user_config) {
        const auto& name = k.first;
        if (meta_device.device_config.find(name) == meta_device.device_config.end() &&
            !ov::util::contains(supported_configKeys, name)) {
            OPENVINO_THROW("Unsupported config key: ", name);
        }
    }
    return meta_device;
}

namespace ov {
namespace autobatch_plugin {
static const ov::Version version = {CI_BUILD_NUMBER, "openvino_auto_batch_plugin"};

OV_DEFINE_PLUGIN_CREATE_FUNCTION(ov::autobatch_plugin::Plugin, version)
}  // namespace autobatch_plugin
}  // namespace ov
