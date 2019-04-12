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

#pragma once
#include "inference_engine.hpp"
#include "ie_ihetero_plugin.hpp"
#include "description_buffer.hpp"
#include "ie_error.hpp"
#include <memory>
#include <string>
#include <map>
#include <vector>
#include <cpp_interfaces/impl/ie_plugin_internal.hpp>

namespace HeteroPlugin {

class Engine : public InferenceEngine::InferencePluginInternal {
public:
    Engine();

    void GetVersion(const InferenceEngine::Version *&versionInfo) noexcept;

    InferenceEngine::ExecutableNetworkInternal::Ptr
    LoadExeNetworkImpl(InferenceEngine::ICNNNetwork &network,
                       const std::map<std::string, std::string> &config) override;
    void SetConfig(const std::map<std::string, std::string> &config) override;

    void SetDeviceLoader(const std::string &device, InferenceEngine::IHeteroDeviceLoader::Ptr pLoader);

    void SetAffinity(InferenceEngine::ICNNNetwork& network, const std::map<std::string, std::string> &config);

    void AddExtension(InferenceEngine::IExtensionPtr extension)override;

    void SetLogCallback(InferenceEngine::IErrorListener &listener) override;
private:
    std::vector<InferenceEngine::IExtensionPtr> _extensions;
    InferenceEngine::MapDeviceLoaders _deviceLoaders;
    InferenceEngine::IErrorListener* error_listener = nullptr;
};

}  // namespace HeteroPlugin
