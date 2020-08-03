// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * \brief inference engine plugin API wrapper, to be used by particular implementors
 * \file ie_plugin_base.hpp
 */

#pragma once

#include <map>
#include <memory>
#include <string>

#include "cpp_interfaces/interface/ie_plugin.hpp"
#include "cpp_interfaces/exception2status.hpp"
#include "description_buffer.hpp"

namespace InferenceEngine {

/**
 * @brief Plugin `noexcept` wrapper which accepts IInferencePluginInternal derived instance which can throw exceptions
 * @ingroup ie_dev_api_plugin_api
 * @tparam T Minimal CPP implementation of IInferencePluginInternal (e.g. InferencePluginInternal)
 */
template <class T>
class PluginBase : public IInferencePlugin {
    class VersionStore : public Version {
        std::string _dsc;
        std::string _buildNumber;

    public:
        explicit VersionStore(const Version& v) {
            _dsc = v.description;
            _buildNumber = v.buildNumber;
            description = _dsc.c_str();
            buildNumber = _buildNumber.c_str();
            apiVersion = v.apiVersion;
        }
    } _version;

    std::shared_ptr<T> _impl;

public:
    /**
     * @brief Constructor with plugin version and actual underlying implementation.
     * @param actualReported version that are to be reported
     * @param impl Underplying implementation of type IInferencePluginInternal
     */
    PluginBase(const Version& actualReported, std::shared_ptr<T> impl): _version(actualReported) {
        if (impl.get() == nullptr) {
            THROW_IE_EXCEPTION << "implementation not defined";
        }
        _impl = impl;
    }

    void SetName(const std::string& pluginName) noexcept override {
        _impl->SetName(pluginName);
    }

    std::string GetName() const noexcept override {
        return _impl->GetName();
    }

    void GetVersion(const Version*& versionInfo) noexcept override {
        versionInfo = &_version;
    }

    StatusCode LoadNetwork(IExecutableNetwork::Ptr& executableNetwork, const ICNNNetwork& network,
                           const std::map<std::string, std::string>& config, ResponseDesc* resp) noexcept override {
        TO_STATUS(_impl->LoadNetwork(executableNetwork, network, config));
    }

    StatusCode AddExtension(InferenceEngine::IExtensionPtr extension,
                            InferenceEngine::ResponseDesc* resp) noexcept override {
        TO_STATUS(_impl->AddExtension(extension));
    }

    StatusCode SetConfig(const std::map<std::string, std::string>& config, ResponseDesc* resp) noexcept override {
        TO_STATUS(_impl->SetConfig(config));
    }

    StatusCode ImportNetwork(IExecutableNetwork::Ptr& ret, const std::string& modelFileName,
                             const std::map<std::string, std::string>& config, ResponseDesc* resp) noexcept override {
        TO_STATUS(ret = _impl->ImportNetwork(modelFileName, config));
    }

    ExecutableNetwork ImportNetwork(std::istream& networkModel,
                                    const std::map<std::string, std::string>& config) override {
        return _impl->ImportNetwork(networkModel, config);
    }

    void Release() noexcept override {
        delete this;
    }

    void QueryNetwork(const ICNNNetwork& network, const std::map<std::string, std::string>& config,
                      QueryNetworkResult& res) const noexcept override {
        TO_STATUSVAR(_impl->QueryNetwork(network, config, res), res.rc, &res.resp);
    }

    void SetCore(ICore* core) noexcept override {
        _impl->SetCore(core);
    }

    const ICore& GetCore() const override {
        IE_ASSERT(nullptr != _impl->GetCore());
        return *_impl->GetCore();
    }

    Parameter GetConfig(const std::string& name, const std::map<std::string, Parameter>& options) const override {
        return _impl->GetConfig(name, options);
    }

    Parameter GetMetric(const std::string& name, const std::map<std::string, Parameter>& options) const override {
        return _impl->GetMetric(name, options);
    }

    RemoteContext::Ptr CreateContext(const ParamMap& params) override {
        return _impl->CreateContext(params);
    }

    RemoteContext::Ptr GetDefaultContext() override {
        return _impl->GetDefaultContext();
    }

    ExecutableNetwork LoadNetwork(const ICNNNetwork& network, const std::map<std::string, std::string>& config,
                                  RemoteContext::Ptr context) override {
        return _impl->LoadNetwork(network, config, context);
    }

    ExecutableNetwork ImportNetwork(std::istream& networkModel,
                                    const RemoteContext::Ptr& context,
                                    const std::map<std::string, std::string>& config) override {
        return _impl->ImportNetwork(networkModel, context, config);
    }

private:
    ~PluginBase() override {}
};

template <class T>
inline IInferencePlugin* make_ie_compatible_plugin(const Version& reported, std::shared_ptr<T> impl) {
    return new PluginBase<T>(reported, impl);
}

}  // namespace InferenceEngine
