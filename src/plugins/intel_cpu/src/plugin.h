// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "exec_network.h"

#include <string>
#include <map>
#include <memory>
#include <functional>

namespace ov {
namespace intel_cpu {

class Engine : public InferenceEngine::IInferencePlugin {
public:
    Engine();
    ~Engine();

    std::shared_ptr<InferenceEngine::IExecutableNetworkInternal>
    LoadExeNetworkImpl(const InferenceEngine::CNNNetwork &network,
                       const std::map<std::string, std::string> &config) override;

    void AddExtension(const InferenceEngine::IExtensionPtr& extension) override;

    void SetConfig(const std::map<std::string, std::string> &config) override;

    InferenceEngine::Parameter GetConfig(const std::string& name, const std::map<std::string, InferenceEngine::Parameter>& options) const override;

    InferenceEngine::Parameter GetMetric(const std::string& name, const std::map<std::string, InferenceEngine::Parameter>& options) const override;

    InferenceEngine::QueryNetworkResult QueryNetwork(const InferenceEngine::CNNNetwork& network,
                                                     const std::map<std::string, std::string>& config) const override;

    InferenceEngine::IExecutableNetworkInternal::Ptr ImportNetwork(std::istream& networkModel,
                                                     const std::map<std::string, std::string>& config) override;

private:
    bool isLegacyAPI() const;

    InferenceEngine::Parameter GetMetricLegacy(const std::string& name, const std::map<std::string, InferenceEngine::Parameter>& options) const;

    InferenceEngine::Parameter GetConfigLegacy(const std::string& name, const std::map<std::string, InferenceEngine::Parameter>& options) const;

    void ApplyPerformanceHints(std::map<std::string, std::string> &config, const std::shared_ptr<ngraph::Function>& ngraphFunc) const;

    struct StreamCfg {
        int num_streams;
        int big_core_streams;          // Number of streams in Performance-core(big core)
        int small_core_streams;        // Number of streams in Efficient-core(small core)
        int threads_per_stream_big;    // Threads per stream in big cores
        int threads_per_stream_small;  // Threads per stream in small cores
        int small_core_offset;
    };
    enum StreamMode { DEFAULT, AGGRESSIVE, LESSAGGRESSIVE };
    StreamCfg GetNumStreams(InferenceEngine::IStreamsExecutor::ThreadBindingType thread_binding_type,
                            int stream_mode,
                            const bool enable_hyper_thread = true) const;

    Config engConfig;
    ExtensionManager::Ptr extensionManager = std::make_shared<ExtensionManager>();
    /* Explicily configured streams have higher priority even than performance hints.
       So track if streams is set explicitly (not auto-configured) */
    bool streamsExplicitlySetForEngine = false;
    const std::string deviceFullName;

    std::shared_ptr<void> specialSetup;
};

}   // namespace intel_cpu
}   // namespace ov
