// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///////////////////////////////////////////////////////////////////////////////////////////////////

#include "plugin.hpp"

#include "compiled_model.hpp"
#include "ie_icore.hpp"
#include "ie_metric_helpers.hpp"
#include "ie_ngraph_utils.hpp"
#include "ie_performance_hints.hpp"
#include "openvino/core/dimension_tracker.hpp"
#include "openvino/pass/manager.hpp"
#include "openvino/runtime/intel_gpu/properties.hpp"
#include "openvino/util/common_util.hpp"
#include "transformations/common_optimizations/dimension_tracking.hpp"
#include "transformations/init_node_info.hpp"
#include "transformations/utils/utils.hpp"

namespace ov {
namespace autobatch_plugin {

std::vector<std::string> supported_configKeys = {CONFIG_KEY(AUTO_BATCH_DEVICE_CONFIG),
                                                 ov::device::priorities.name(),
                                                 CONFIG_KEY(AUTO_BATCH_TIMEOUT),
                                                 CONFIG_KEY(CACHE_DIR)};
namespace {

std::map<std::string, std::string> mergeConfigs(std::map<std::string, std::string> config,
                                                const std::map<std::string, std::string>& user_config) {
    for (auto&& kvp : user_config) {
        config[kvp.first] = kvp.second;
    }
    return config;
}

}  // namespace

DeviceInformation Plugin::ParseBatchDevice(const std::string& deviceWithBatch) {
    auto&& d = deviceWithBatch;
    auto openingBracket = d.find_first_of('(');
    auto closingBracket = d.find_first_of(')', openingBracket);
    auto deviceName = d.substr(0, openingBracket);

    int batch = 0;
    if (closingBracket != std::string::npos && openingBracket < closingBracket) {
        batch = std::stol(d.substr(openingBracket + 1, closingBracket - 1));

        if (batch <= 0) {
            IE_THROW() << "Batch value for '" << deviceName << "' must be > 0, while " << batch << "is passed";
        }
    }
    return {deviceName, {{}}, batch};
}

DeviceInformation Plugin::ParseMetaDevice(const std::string& devicesBatchCfg,
                                          const std::map<std::string, std::string>& user_config) const {
    auto metaDevice = ParseBatchDevice(devicesBatchCfg);
    metaDevice.config = GetCore()->GetSupportedConfig(metaDevice.device_name, user_config);

    // check that no irrelevant config-keys left
    for (const auto& k : user_config) {
        const auto& name = k.first;
        if (metaDevice.config.find(name) == metaDevice.config.end() &&
            !ov::util::contains(supported_configKeys, name)) {
            IE_THROW() << "Unsupported config key: " << name;
        }
    }
    return metaDevice;
}

InferenceEngine::RemoteContext::Ptr Plugin::CreateContext(const InferenceEngine::ParamMap& remote_properties) {
    auto cfg = remote_properties;
    auto it = cfg.find(CONFIG_KEY(AUTO_BATCH_DEVICE_CONFIG));
    if (it == cfg.end())
        it = cfg.find(ov::device::priorities.name());
    if (it == cfg.end())
        IE_THROW() << "Value for KEY_AUTO_BATCH_DEVICE_CONFIG is not set";

    auto val = it->second.as<std::string>();
    auto core = GetCore();
    if (!core)
        return nullptr;
    auto metaDevice = ParseMetaDevice(val, std::map<std::string, std::string>());
    cfg.erase(it);
    return core->CreateContext(metaDevice.device_name, cfg);
}

InferenceEngine::Parameter Plugin::GetConfig(
    const std::string& name,
    const std::map<std::string, InferenceEngine::Parameter>& user_options) const {
    if (supported_configKeys.end() != std::find(supported_configKeys.begin(), supported_configKeys.end(), name)) {
        auto it = _config.find(name);
        if (it == _config.end()) {
            IE_THROW() << "Value for " << name << " is not set";
        } else {
            return {it->second};
        }
    } else {
        IE_THROW() << "Unsupported config key: " << name;
    }
}

void Plugin::CheckConfig(const std::map<std::string, std::string>& user_config) {
    for (auto&& kvp : user_config) {
        const auto name = kvp.first;
        const auto val = kvp.second;
        if (supported_configKeys.end() == std::find(supported_configKeys.begin(), supported_configKeys.end(), name))
            IE_THROW() << "Unsupported config key: " << name;
        if (name == CONFIG_KEY(AUTO_BATCH_DEVICE_CONFIG) || name == ov::device::priorities.name()) {
            ParseBatchDevice(val);
        } else if (name == CONFIG_KEY(AUTO_BATCH_TIMEOUT)) {
            try {
                auto t = std::stoi(val);
                if (t < 0)
                    IE_THROW(ParameterMismatch);
            } catch (const std::exception&) {
                IE_THROW(ParameterMismatch)
                    << " Expecting unsigned int value for " << CONFIG_KEY(AUTO_BATCH_TIMEOUT) << " got " << val;
            }
        }
    }
}

void Plugin::SetConfig(const std::map<std::string, std::string>& user_config) {
    CheckConfig(user_config);
    for (auto&& kvp : user_config) {
        _config[kvp.first] = kvp.second;
    }
}

static const InferenceEngine::Version version = {{2, 1}, CI_BUILD_NUMBER, "AutoBatchPlugin"};
IE_DEFINE_PLUGIN_CREATE_FUNCTION(Plugin, version)

Plugin::Plugin() {
    _pluginName = "BATCH";
    _config[CONFIG_KEY(AUTO_BATCH_TIMEOUT)] = "1000";  // default value, in ms
}

InferenceEngine::Parameter Plugin::GetMetric(
    const std::string& name,
    const std::map<std::string, InferenceEngine::Parameter>& user_options) const {
    if (name == METRIC_KEY(SUPPORTED_METRICS)) {
        std::vector<std::string> metrics;
        metrics.push_back(METRIC_KEY(SUPPORTED_METRICS));
        metrics.push_back(METRIC_KEY(FULL_DEVICE_NAME));
        metrics.push_back(METRIC_KEY(SUPPORTED_CONFIG_KEYS));
        IE_SET_METRIC_RETURN(SUPPORTED_METRICS, metrics);
    } else if (name == METRIC_KEY(FULL_DEVICE_NAME)) {
        IE_SET_METRIC_RETURN(FULL_DEVICE_NAME, _pluginName);
    } else if (name == METRIC_KEY(SUPPORTED_CONFIG_KEYS)) {
        IE_SET_METRIC_RETURN(SUPPORTED_CONFIG_KEYS, supported_configKeys);
    } else {
        IE_THROW(NotFound) << "Unsupported metric key " << name;
    }
}

InferenceEngine::IExecutableNetworkInternal::Ptr Plugin::LoadExeNetworkImpl(
    const InferenceEngine::CNNNetwork& network,
    const std::map<std::string, std::string>& user_config) {
    return LoadNetworkImpl(network, nullptr, user_config);
}

InferenceEngine::IExecutableNetworkInternal::Ptr Plugin::LoadNetworkImpl(
    const InferenceEngine::CNNNetwork& network,
    const std::shared_ptr<InferenceEngine::RemoteContext> ctx,
    const std::map<std::string, std::string>& user_config) {
    auto core = GetCore();
    if (core == nullptr) {
        IE_THROW() << "Please, work with Auto-Batching device via InferencEngine::Core object";
    }
    auto fullConfig = mergeConfigs(_config, user_config);
    auto device_batch = fullConfig.find(CONFIG_KEY(AUTO_BATCH_DEVICE_CONFIG));
    if (device_batch == fullConfig.end())
        device_batch = fullConfig.find(ov::device::priorities.name());
    if (device_batch == fullConfig.end()) {
        IE_THROW() << "KEY_AUTO_BATCH key is not set for BATCH device";
    }
    auto metaDevice = ParseMetaDevice(device_batch->second, user_config);
    const auto& deviceName = metaDevice.device_name;
    const auto& deviceConfig = metaDevice.config;
    auto deviceConfigNoAutoBatch = deviceConfig;
    // avoid recursive auto-batching
    deviceConfigNoAutoBatch[CONFIG_KEY(ALLOW_AUTO_BATCHING)] = CONFIG_VALUE(NO);

    std::set<std::string> batched_inputs;
    std::set<std::string> batched_outputs;
    // check that the auto-batching is applicable in general
    try {
        // if applicable, the Auto-Batching is implicitly enabled via the performance hints
        const auto tput = CONFIG_VALUE(THROUGHPUT);
        const bool bTputInPlg = core->GetConfig(deviceName, CONFIG_KEY(PERFORMANCE_HINT)).as<std::string>() == tput;
        const auto& mode = deviceConfig.find(CONFIG_KEY(PERFORMANCE_HINT));
        const bool bTputInLoadCfg = (mode != deviceConfig.end() && mode->second == tput);
        // if the auto-batching is enabled implicitly, check the dims carefully, to avoid outstanding failures
        const bool check_dims = (bTputInPlg || bTputInLoadCfg);
        InferenceEngine::CNNNetwork clonedNetwork(InferenceEngine::details::cloneNetwork(network));
        auto function = clonedNetwork.getFunction();
        // find the batch dim
        ov::pass::Manager m;
        m.register_pass<ov::pass::InitNodeInfo>();
        m.register_pass<ov::pass::FindBatch>(false, check_dims);
        m.run_passes(function);
        // do not reshape/re-batch originally batched networks and when there are no inputs with the N* layouts
        // input(s) should have the batch dim as the first dim (current limitation of the auto-batching impl)
        const auto& params = function->get_parameters();
        for (size_t input_id = 0; input_id < params.size(); input_id++) {
            const auto& input = params[input_id];
            const auto& shape = input->get_partial_shape();
            // currently no plugin support batched execution for dynamic networks
            if (shape.is_dynamic())
                IE_THROW(NotImplemented) << "Auto-batching does not support dynamic networks!";
            // check the batch dim: either 0th (and the original batch size of 1) or none
            if (shape.size() && ov::DimensionTracker::get_label(shape[0])) {
                const auto& static_shape = input->get_shape();
                if (static_shape[0] != 1)
                    IE_THROW(NotImplemented) << "Auto-batching does not reshape/re-batch originally batched networks!";
                batched_inputs.insert(
                    ov::op::util::get_ie_output_name(params[input_id]->output(0)));  // batched dim for the input
            } else {
                // if the 0-th dim is not for the batch, then we support only the case when NONE dimension is batch
                for (size_t s = 1; s < shape.size(); s++)
                    if (ov::DimensionTracker::get_label(shape[s]))
                        IE_THROW(NotImplemented)
                            << "Auto-batching operates only networks with inputs/outputs batched by 0th dimension";
            }
        }
        const auto& results = function->get_results();
        for (size_t output_id = 0; output_id < results.size(); output_id++) {
            const auto& output = results[output_id];
            const auto& shape = output->get_output_partial_shape(0);
            if (shape.is_dynamic())
                IE_THROW(NotImplemented) << "Auto-batching does not support dynamic networks!";
            // check the batch dim: either 0th (and the original batch size of 1) or none
            if (shape.size() && ov::DimensionTracker::get_label(shape[0])) {
                if (shape[0] != 1)
                    IE_THROW(NotImplemented) << "Auto-batching does not reshape/re-batch originally batched networks!";
                const auto& node = output->input_value(0);
                batched_outputs.insert(
                    ov::op::util::get_ie_output_name(ov::Output<const ov::Node>(node.get_node(), node.get_index())));
            } else {
                // if the 0-th dim is not for the batch, then we support only the case when NONE dimension is batch
                for (size_t s = 1; s < shape.size(); s++)
                    if (ov::DimensionTracker::get_label(shape[s]))
                        IE_THROW(NotImplemented)
                            << "Auto-batching operates only networks with outputs batched by 0th dimension";
            }
        }
        if (!batched_inputs.size() || !batched_outputs.size())
            IE_THROW(NotImplemented)
                << "Auto-batching supports only networks with inputs/outputs featuring batched dim!";
    } catch (const InferenceEngine::Exception&) {
        metaDevice.batch_for_device = 1;
    }

    if (!metaDevice.batch_for_device) {
        unsigned int requests = 0;
        // batch size is not set explicitly via device name e.g. BATCH:GPU(4)
        // let's query the optimal batch size
        std::map<std::string, InferenceEngine::Parameter> options;
        options["MODEL_PTR"] = std::const_pointer_cast<ngraph::Function>(network.getFunction());
        auto optBatchSize = core->GetMetric(deviceName, METRIC_KEY(OPTIMAL_BATCH_SIZE), options).as<unsigned int>();
        auto res = core->GetConfig(deviceName, CONFIG_KEY(PERFORMANCE_HINT_NUM_REQUESTS)).as<std::string>();
        requests = InferenceEngine::PerfHintsConfig::CheckPerformanceHintRequestValue(res);
        const auto& reqs = user_config.find(CONFIG_KEY(PERFORMANCE_HINT_NUM_REQUESTS));
        if (reqs != user_config.end())
            requests = static_cast<unsigned int>(
                InferenceEngine::PerfHintsConfig::CheckPerformanceHintRequestValue(reqs->second));
        if (requests)
            optBatchSize = std::max(1u, std::min(requests, optBatchSize));
        if (optBatchSize > 2)  // batching is usually in-efficient for batch<4 (as batch1 kernels are heavily optimized)
            metaDevice.batch_for_device = optBatchSize;
        else
            metaDevice.batch_for_device = 1;
    }

    auto report_footprint = [](std::shared_ptr<InferenceEngine::ICore> pCore, std::string device) -> size_t {
        size_t footprint = 0;
        // TODO: use the per-network metric (22.2) rather than plugin-level
        auto stats =
            pCore->GetMetric(device, ov::intel_gpu::memory_statistics.name()).as<std::map<std::string, uint64_t>>();
        for (const auto& s : stats)
            footprint += s.second;
        return footprint;
    };

    size_t batch1_footprint = 0;
    if (deviceName.find("GPU") != std::string::npos)
        batch1_footprint = report_footprint(core, deviceName);
    auto executableNetworkWithoutBatch = ctx ? core->LoadNetwork(network, ctx, deviceConfigNoAutoBatch)
                                             : core->LoadNetwork(network, deviceName, deviceConfigNoAutoBatch);
    if (deviceName.find("GPU") != std::string::npos) {
        batch1_footprint = report_footprint(core, deviceName) - batch1_footprint;
        if (batch1_footprint) {
            const auto total_mem =
                GetCore()->GetMetric(deviceName, GPU_METRIC_KEY(DEVICE_TOTAL_MEM_SIZE)).as<uint64_t>();
            const int estimated_batch = static_cast<int>((total_mem - batch1_footprint) / batch1_footprint);
            int closest = static_cast<int>(pow(2, floor(std::log(estimated_batch) / std::log(2))));
            closest = std::max(1, closest);
            metaDevice.batch_for_device = std::min(metaDevice.batch_for_device, closest);
        }
    }
    // auto-batch settings
    std::unordered_map<std::string, InferenceEngine::Parameter> networkConfig;
    for (const auto& c : fullConfig) {
        if (supported_configKeys.end() != std::find(supported_configKeys.begin(), supported_configKeys.end(), c.first))
            networkConfig.insert(c);
    }

    InferenceEngine::SoExecutableNetworkInternal executableNetworkWithBatch;
    if (metaDevice.batch_for_device > 1 && batched_inputs.size()) {
        try {
            InferenceEngine::CNNNetwork reshaped(InferenceEngine::details::cloneNetwork(network));
            InferenceEngine::ICNNNetwork::InputShapes shapes = reshaped.getInputShapes();
            for (const auto& input : batched_inputs)
                shapes[input][0] = metaDevice.batch_for_device;
            reshaped.reshape(shapes);
            executableNetworkWithBatch = ctx ? core->LoadNetwork(reshaped, ctx, deviceConfigNoAutoBatch)
                                             : core->LoadNetwork(reshaped, deviceName, deviceConfigNoAutoBatch);
        } catch (const InferenceEngine::Exception&) {
            metaDevice.batch_for_device = 1;
        }
    }

    return std::make_shared<CompiledModel>(executableNetworkWithBatch,
                                           executableNetworkWithoutBatch,
                                           metaDevice,
                                           networkConfig,
                                           batched_inputs,
                                           batched_outputs);
}

InferenceEngine::IExecutableNetworkInternal::Ptr Plugin::LoadExeNetworkImpl(
    const InferenceEngine::CNNNetwork& network,
    const std::shared_ptr<InferenceEngine::RemoteContext>& context,
    const std::map<std::string, std::string>& user_config) {
    return LoadNetworkImpl(network, context, user_config);
}

InferenceEngine::QueryNetworkResult Plugin::QueryNetwork(const InferenceEngine::CNNNetwork& network,
                                                         const std::map<std::string, std::string>& user_config) const {
    auto core = GetCore();
    if (!core)
        return InferenceEngine::QueryNetworkResult();
    auto cfg = user_config;
    for (const auto& c : cfg) {
        if (c.first == CONFIG_KEY(AUTO_BATCH_DEVICE_CONFIG) || c.first == ov::device::priorities.name()) {
            auto val = c.second;
            cfg.erase(c.first);
            auto metaDevice = ParseMetaDevice(val, cfg);
            return core->QueryNetwork(network, metaDevice.device_name, cfg);
        }
    }
    IE_THROW() << "Value for KEY_AUTO_BATCH_DEVICE_CONFIG is not set";
}
}  // namespace autobatch_plugin
}  // namespace ov
