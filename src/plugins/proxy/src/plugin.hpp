// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cpp_interfaces/impl/ie_executable_network_thread_safe_default.hpp>
#include <cpp_interfaces/interface/ie_iplugin_internal.hpp>

namespace ov {
namespace proxy {

class Plugin : public InferenceEngine::IInferencePlugin {
public:
    using Ptr = std::shared_ptr<Plugin>;

    Plugin();
    ~Plugin();

    void SetConfig(const std::map<std::string, std::string>& config) override;
    InferenceEngine::QueryNetworkResult QueryNetwork(const InferenceEngine::CNNNetwork& network,
                                                     const std::map<std::string, std::string>& config) const override;
    InferenceEngine::IExecutableNetworkInternal::Ptr LoadExeNetworkImpl(
        const InferenceEngine::CNNNetwork& network,
        const std::map<std::string, std::string>& config) override;
    void AddExtension(const std::shared_ptr<InferenceEngine::IExtension>& extension) override;
    InferenceEngine::Parameter GetConfig(
        const std::string& name,
        const std::map<std::string, InferenceEngine::Parameter>& options) const override;
    InferenceEngine::Parameter GetMetric(
        const std::string& name,
        const std::map<std::string, InferenceEngine::Parameter>& options) const override;
    InferenceEngine::IExecutableNetworkInternal::Ptr ImportNetwork(
        std::istream& model,
        const std::map<std::string, std::string>& config) override;

private:
    std::vector<std::pair<std::string, std::vector<std::string>>> get_hidden_devices() const;
    std::string get_fallback_device(size_t idx) const;
    std::vector<std::string> get_primary_devices() const;
    std::string get_primary_device(size_t idx) const;
    size_t get_device_from_config(const std::map<std::string, std::string>& config) const;
};

}  // namespace proxy
}  // namespace ov
