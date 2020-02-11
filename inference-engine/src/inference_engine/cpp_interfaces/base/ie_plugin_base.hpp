// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * \brief inference engine plugin API wrapper, to be used by particular implementors
 * \file ie_plugin_base.hpp
 */

#pragma once

#include <ie_plugin.hpp>
#include <map>
#include <memory>
#include <string>

#include "cpp_interfaces/base/ie_inference_plugin_api.hpp"
#include "cpp_interfaces/exception2status.hpp"
#include "description_buffer.hpp"
#include "ie_common.h"

namespace InferenceEngine {

IE_SUPPRESS_DEPRECATED_START

/**
 * @brief cpp interface for plugin, to avoid dll boundaries ,and simplify internal development
 * @tparam T Minimal CPP implementation of IInferencePlugin (e.g. IInferencePluginInternal)
 * @details can be used to create external wrapper too
 */
template <class T>
class PluginBase : public IInferencePluginAPI, public IInferencePlugin {
protected:
    IE_SUPPRESS_DEPRECATED_END

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
     *
     * @param actualReported version that are to be reported
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

    /**
     * @brief return plugin's version information
     * @param versionInfo pointer to version info, will be set by plugin
     */
    void GetVersion(const Version*& versionInfo) noexcept override {
        versionInfo = &_version;
    }

    void SetLogCallback(IErrorListener& listener) noexcept override {
        NO_EXCEPT_CALL_RETURN_VOID(_impl->SetLogCallback(listener));
    }

    StatusCode LoadNetwork(IExecutableNetwork::Ptr& executableNetwork, ICNNNetwork& network,
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

    ExecutableNetwork LoadNetwork(ICNNNetwork& network, const std::map<std::string, std::string>& config,
                                  RemoteContext::Ptr context) override {
        return _impl->LoadNetwork(network, config, context);
    }

private:
    ~PluginBase() override {}
};

IE_SUPPRESS_DEPRECATED_START

template <class T>
inline IInferencePlugin* make_ie_compatible_plugin(const Version& reported, std::shared_ptr<T> impl) {
    return new PluginBase<T>(reported, impl);
}

IE_SUPPRESS_DEPRECATED_END

}  // namespace InferenceEngine
