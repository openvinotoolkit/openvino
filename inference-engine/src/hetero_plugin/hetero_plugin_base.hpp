//
// Copyright (C) 2018-2019 Intel Corporation.
//
// This software and the related documents are Intel copyrighted materials,
// and your use of them is governed by the express license under which they
// were provided to you (End User License Agreement for the Intel(R) Software
// Development Products (Version May 2017)). Unless the License provides
// otherwise, you may not use, modify, copy, publish, distribute, disclose or
// transmit this software or the related documents without Intel's prior
// written permission.
//
// This software and the related documents are provided as is, with no
// express or implied warranties, other than those that are expressly
// stated in the License.
//

/**
 * \brief inference engine plugin API wrapper, to be used by particular implementors
 * \file ie_plugin_base.hpp
 */

#pragma once

#include <memory>
#include <map>
#include <string>
#include <ie_plugin.hpp>
#include <ie_ihetero_plugin.hpp>
#include "description_buffer.hpp"
#include "cpp_interfaces/exception2status.hpp"

namespace InferenceEngine {

/**
 * @brief cpp interface for plugin, to avoid dll boundaries ,and simplify internal development
 * @tparam T Minimal CPP implementation of IInferencePlugin (e.g. IInferencePluginInternal)
 * @details can be used to create external wrapper too
 */
template<class T>
class HeteroPluginBase : public IHeteroInferencePlugin {
protected:
    class VersionStore : public Version {
        std::string _dsc;
        std::string _buildNumber;
    public:
        explicit VersionStore(const Version &v) {
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
    HeteroPluginBase(const Version &actualReported, std::shared_ptr<T> impl) : _version(actualReported) {
        if (impl.get() == nullptr) {
            THROW_IE_EXCEPTION << "implementation not defined";
        }
        _impl = impl;
    }

    /**
     * @brief return plugin's version information
     * @param versionInfo pointer to version info, will be set by plugin
     */
    void GetVersion(const Version *&versionInfo) noexcept override {
        versionInfo = &_version;
    }

    void SetLogCallback(IErrorListener &listener) noexcept override {
        NO_EXCEPT_CALL_RETURN_VOID(_impl->SetLogCallback(listener));
    }

    StatusCode LoadNetwork(ICNNNetwork &network, ResponseDesc *resp) noexcept override {
        TO_STATUS(_impl->LoadNetwork(network));
    }

    StatusCode LoadNetwork(IExecutableNetwork::Ptr &executableNetwork,
                           ICNNNetwork &network,
                           const std::map<std::string, std::string> &config,
                           ResponseDesc *resp)noexcept override {
        TO_STATUS(_impl->LoadNetwork(executableNetwork, network, config));
    }

    StatusCode Infer(const Blob &input, Blob &result, ResponseDesc *resp) noexcept override {
        TO_STATUS(_impl->Infer(input, result));
    }

    StatusCode Infer(const BlobMap &input, BlobMap &result, ResponseDesc *resp)noexcept override {
        TO_STATUS(_impl->Infer(input, result));
    }

    StatusCode GetPerformanceCounts(std::map<std::string,
            InferenceEngineProfileInfo> &perfMap, ResponseDesc *resp) const noexcept override {
        TO_STATUS(_impl->GetPerformanceCounts(perfMap));
    }

    StatusCode AddExtension(InferenceEngine::IExtensionPtr extension,
                            InferenceEngine::ResponseDesc *resp) noexcept override {
        TO_STATUS(_impl->AddExtension(extension));
    }

    StatusCode SetConfig(const std::map<std::string, std::string> &config, ResponseDesc *resp) noexcept override {
        TO_STATUS(_impl->SetConfig(config));
    }

    StatusCode ImportNetwork(IExecutableNetwork::Ptr &ret, const std::string &modelFileName,
                             const std::map<std::string, std::string> &config, ResponseDesc *resp) noexcept override {
        TO_STATUS(ret = _impl->ImportNetwork(modelFileName, config));
    }

    void Release() noexcept override {
        delete this;
    }

    void SetDeviceLoader(const std::string &device, IHeteroDeviceLoader::Ptr loader)noexcept override {
        _impl->SetDeviceLoader(device, loader);
    }

    StatusCode SetAffinity(
        ICNNNetwork& network,
        const std::map<std::string, std::string> &config,
        ResponseDesc *resp) noexcept override {
        TO_STATUS(_impl->SetAffinity(network, config));
    }


 private:
    ~HeteroPluginBase() = default;
};

}  // namespace InferenceEngine
