// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "exec_network.h"
#include "cpu_streams_calculation.hpp"

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

    void GetPerformanceStreams(Config &config, const std::shared_ptr<ngraph::Function>& ngraphFunc);

    void CalculateStreams(Config& conf, const std::shared_ptr<ngraph::Function>& ngraphFunc, bool imported = false);

    StreamCfg GetNumStreams(InferenceEngine::IStreamsExecutor::ThreadBindingType thread_binding_type,
                            int stream_mode,
                            const bool enable_hyper_thread = true) const;

    //Initialize Xbyak::util::Cpu object on the specified core type
    void InitCpuInfo(const std::map<std::string, std::string> &config, Config::ModelType modelType);

    Config engConfig;
    ExtensionManager::Ptr extensionManager = std::make_shared<ExtensionManager>();
    /* Explicily configured streams have higher priority than performance hints.
       So track if streams is set explicitly (not auto-configured) */
    bool streamsExplicitlySetForEngine = false;
    const std::string deviceFullName;

    std::shared_ptr<void> specialSetup;

#if defined(OV_CPU_WITH_ACL)
    struct SchedulerGuard {
        SchedulerGuard();
        ~SchedulerGuard();
        static std::shared_ptr<SchedulerGuard> instance();
        static std::mutex mutex;
        // separate mutex for saving ACLScheduler state in destructor
        mutable std::mutex dest_mutex;
        static std::weak_ptr<SchedulerGuard> ptr;
    };

    std::shared_ptr<SchedulerGuard> scheduler_guard;
#endif
};

}   // namespace intel_cpu
}   // namespace ov
