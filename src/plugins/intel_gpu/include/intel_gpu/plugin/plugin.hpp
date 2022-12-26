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

class Plugin : public InferenceEngine::IInferencePlugin {
    struct impl;
    std::shared_ptr<impl> _impl;
    bool streamsSet = false;
    bool throttlingSet = false;
    bool isModelCachingEnabled = false;
    std::string default_device_id = "0";

    // key: device_id, value: cldnn device
    std::map<std::string, cldnn::device::ptr> device_map;
    // key: cldnn context, value: memory statistics
    mutable std::map<RemoteContextImpl::Ptr, std::map<std::string, uint64_t>> statistics_map;
    mutable std::mutex engine_mutex;

    mutable std::map<std::string, RemoteCLContext::Ptr> m_defaultContexts;

    cldnn::device_info GetDeviceInfo(const std::map<std::string, std::string> &config) const;
    InferenceEngine::CNNNetwork CloneAndTransformNetwork(const InferenceEngine::CNNNetwork& network,
                                                         const Config& config) const;
    void TransformNetwork(std::shared_ptr<ov::Model>& model, const Config& config) const;
    std::map<std::string, std::string> ConvertPerfHintsToConfig(const std::map<std::string, std::string>& network_config, const Config& plugin_config) const;
    void RegisterPrimitives();
    void UpdateConfig(Config& conf, const InferenceEngine::CNNNetwork &network, const std::map<std::string, std::string> &params) const;
    void UpdateStatistics(const RemoteContextImpl::Ptr& context) const;
    std::string get_device_id_from_config(const std::map<std::string, std::string>& config) const;
    RemoteCLContext::Ptr get_default_context(const Config& config) const;

public:
    Plugin();

    InferenceEngine::IExecutableNetworkInternal::Ptr LoadExeNetworkImpl(const InferenceEngine::CNNNetwork &network,
                                                                        const std::map<std::string, std::string> &config) override;

    InferenceEngine::IExecutableNetworkInternal::Ptr LoadExeNetworkImpl(const InferenceEngine::CNNNetwork &network,
                                                                        const std::shared_ptr<InferenceEngine::RemoteContext> &context,
                                                                        const std::map<std::string, std::string> &config) override;

    void SetConfig(const std::map<std::string, std::string> &config) override;
    InferenceEngine::Parameter GetConfig(const std::string& name, const std::map<std::string, InferenceEngine::Parameter>& options) const override;
    InferenceEngine::Parameter GetMetric(const std::string& name, const std::map<std::string, InferenceEngine::Parameter>& options) const override;
    InferenceEngine::QueryNetworkResult QueryNetwork(const InferenceEngine::CNNNetwork& network,
                                                     const std::map<std::string, std::string>& config) const override;
    InferenceEngine::IExecutableNetworkInternal::Ptr ImportNetwork(std::istream& networkModel,
                                                     const std::map<std::string, std::string>& config) override;

    std::shared_ptr<InferenceEngine::RemoteContext> CreateContext(const InferenceEngine::ParamMap& params) override;
    std::shared_ptr<InferenceEngine::RemoteContext> GetDefaultContext(const InferenceEngine::ParamMap& params) override;
};

}  // namespace intel_gpu
}  // namespace ov
