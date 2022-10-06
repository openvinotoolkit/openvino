// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <map>
#include <string>
#include <memory>
#include "intel_gpu/runtime/engine.hpp"
#include <cpp_interfaces/interface/ie_iplugin_internal.hpp>
#include <cpp_interfaces/interface/ie_iexecutable_network_internal.hpp>
#include "intel_gpu/plugin/remote_context.hpp"

namespace ov {
namespace intel_gpu {

using CustomLayerPtr = std::shared_ptr<class CustomLayer>;

class Plugin : public InferenceEngine::IInferencePlugin,
               public InferenceEngine::gpu::details::param_map_obj_getter {
    struct impl;
    std::shared_ptr<impl> _impl;
    bool streamsSet = false;
    bool throttlingSet = false;

    // key: device_id, value: cldnn device
    std::map<std::string, cldnn::device::ptr> device_map;
    // key: cldnn context, value: memory statistics
    mutable std::map<RemoteCLContext::Ptr, std::map<std::string, uint64_t>> statistics_map;
    mutable std::mutex engine_mutex;

    mutable RemoteCLContext::Ptr m_defaultContext;

    cldnn::device_info GetDeviceInfo(const std::map<std::string, std::string> &config) const;
    InferenceEngine::CNNNetwork CloneAndTransformNetwork(const InferenceEngine::CNNNetwork& network,
                                                         const Config& config) const;
    void TransformNetwork(std::shared_ptr<ov::Model>& model, const Config& config) const;
    std::map<std::string, std::string> ConvertPerfHintsToConfig(const std::map<std::string, std::string>& network_config,
                                                                const Config& plugin_config) const;

    void RegisterPrimitives();
    void UpdateConfig(Config& conf, const InferenceEngine::CNNNetwork &network, const std::map<std::string, std::string> &params) const;
    void UpdateStatistics(const RemoteCLContext::Ptr& context) const;
public:
    Plugin();

    InferenceEngine::IExecutableNetworkInternal::Ptr LoadExeNetworkImpl(const InferenceEngine::CNNNetwork &network,
                                                                        const std::map<std::string, std::string> &config) override;

    InferenceEngine::IExecutableNetworkInternal::Ptr LoadExeNetworkImpl(const InferenceEngine::CNNNetwork &network,
                                                                        const std::shared_ptr<InferenceEngine::RemoteContext> &context,
                                                                        const std::map<std::string, std::string> &config) override;

    void SetConfig(const std::map<std::string, std::string> &config) override;
    std::string GetDeviceIDFromConfig(const std::map<std::string, std::string>& config) const;
    InferenceEngine::Parameter GetConfig(const std::string& name, const std::map<std::string, InferenceEngine::Parameter>& options) const override;
    InferenceEngine::Parameter GetMetric(const std::string& name, const std::map<std::string, InferenceEngine::Parameter>& options) const override;
    InferenceEngine::QueryNetworkResult QueryNetwork(const InferenceEngine::CNNNetwork& network,
                                                     const std::map<std::string, std::string>& config) const override;

    std::shared_ptr<InferenceEngine::RemoteContext> CreateContext(const InferenceEngine::ParamMap& params) override;
    std::shared_ptr<InferenceEngine::RemoteContext> GetDefaultContext(const InferenceEngine::ParamMap& params) override;

    struct PluginParams {
        cldnn::queue_types queue_type;
        cldnn::engine_types engine_type;
        cldnn::runtime_types runtime_type;
        bool use_unified_shared_memory;
        InferenceEngine::ITaskExecutor::Ptr task_executor;
    };

    static PluginParams GetParams(const Config& config, const cldnn::device::ptr& dev,
                                  InferenceEngine::gpu_handle_param external_queue = nullptr) {
        PluginParams params;
        params.engine_type = cldnn::engine_types::ocl;
        params.runtime_type = cldnn::runtime_types::ocl;
        if (external_queue) {
            params.queue_type = cldnn::stream::detect_queue_type(params.engine_type, external_queue);
        } else if (dev->get_info().supports_immad) {
            params.queue_type = cldnn::queue_types::in_order;
        } else {
            params.queue_type = cldnn::queue_types::out_of_order;
        }
        params.use_unified_shared_memory = true;
        params.task_executor = std::make_shared<InferenceEngine::CPUStreamsExecutor>(config.task_exec_config);
        return params;
    }
};

}  // namespace intel_gpu
}  // namespace ov
