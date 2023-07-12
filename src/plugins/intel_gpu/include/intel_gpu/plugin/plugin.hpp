// Copyright (C) 2018-2023 Intel Corporation
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

class Plugin : public InferenceEngine::IInferencePlugin {
    struct impl;
    std::shared_ptr<impl> _impl;

    std::string default_device_id = "0";
    // key: device_id, value: cldnn device
    std::map<std::string, cldnn::device::ptr> device_map;
    std::map<std::string, ExecutionConfig> m_configs_map;

    mutable std::map<std::string, RemoteCLContext::Ptr> m_default_contexts;
    mutable std::once_flag m_default_contexts_once;

    std::map<std::string, RemoteCLContext::Ptr> get_default_contexts() const;

    InferenceEngine::CNNNetwork clone_and_transform_model(const InferenceEngine::CNNNetwork& network,
                                                          const ExecutionConfig& config) const;
    void transform_model(std::shared_ptr<ov::Model>& model, const ExecutionConfig& config) const;
    void register_primitives();
    std::string get_device_id_from_config(const std::map<std::string, std::string>& config) const;
    std::string get_device_id(const std::map<std::string, std::string>& config) const;
    RemoteCLContext::Ptr get_default_context(const std::string& device_id) const;

    std::vector<ov::PropertyName> get_supported_properties() const;
    std::vector<ov::PropertyName> get_supported_internal_properties() const;
    std::vector<std::string> get_device_capabilities(const cldnn::device_info& info) const;
    uint32_t get_optimal_batch_size(const std::map<std::string, InferenceEngine::Parameter>& options) const;
    uint32_t get_max_batch_size(const std::map<std::string, InferenceEngine::Parameter>& options) const;

    ov::AnyMap preprocess_config(const std::map<std::string, std::string>& orig_config) const;

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
    InferenceEngine::IExecutableNetworkInternal::Ptr ImportNetwork(std::istream& networkModel,
                                                                   const std::shared_ptr<InferenceEngine::RemoteContext>& context,
                                                                   const std::map<std::string, std::string>& config) override;

    std::shared_ptr<InferenceEngine::RemoteContext> CreateContext(const InferenceEngine::ParamMap& params) override;
    std::shared_ptr<InferenceEngine::RemoteContext> GetDefaultContext(const InferenceEngine::ParamMap& params) override;
};

}  // namespace intel_gpu
}  // namespace ov
