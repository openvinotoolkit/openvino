// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <map>
#include <string>
#include <memory>
#include <cldnn/runtime/engine.hpp>
#include <cpp_interfaces/interface/ie_iplugin_internal.hpp>
#include <cpp_interfaces/interface/ie_iexecutable_network_internal.hpp>
#include "cldnn_remote_context.h"

namespace CLDNNPlugin {

using CLDNNCustomLayerPtr = std::shared_ptr<class CLDNNCustomLayer>;

class clDNNEngine : public InferenceEngine::IInferencePlugin,
                    public InferenceEngine::gpu::details::param_map_obj_getter {
    struct impl;
    std::shared_ptr<impl> _impl;
    bool streamsSet = false;
    bool throttlingSet = false;

    // key: device_id, value: cldnn device
    std::map<std::string, cldnn::device::ptr> device_map;
    // key : cldnn context, value: memory statistics
    std::map<CLDNNRemoteCLContext::Ptr, std::map<std::string, uint64_t>> statistics_map;
    std::mutex engine_mutex;

    mutable CLDNNRemoteCLContext::Ptr m_defaultContext;

    cldnn::device_info GetDeviceInfo(const std::map<std::string, std::string> &config) const;
    InferenceEngine::CNNNetwork CloneAndTransformNetwork(const InferenceEngine::CNNNetwork& network,
                                                         const CLDNNPlugin::Config& config) const;

    std::map<std::string, std::string> ConvertPerfHintsToConfig(const std::map<std::string, std::string>& network_config,
                                                               const CLDNNPlugin::Config& plugin_config) const;

    void RegisterPrimitives();
    void UpdateConfig(Config& conf, const InferenceEngine::CNNNetwork &network, const std::map<std::string, std::string> &params) const;
    void UpdateStatistics(CLDNNRemoteCLContext::Ptr context);
public:
    clDNNEngine();

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
};

};  // namespace CLDNNPlugin
