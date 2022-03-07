// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///////////////////////////////////////////////////////////////////////////////////////////////////
#pragma once

#include "multi_schedule.hpp"

#ifdef  MULTIUNITTEST
#define MOCKTESTMACRO virtual
#define MultiDevicePlugin MockMultiDevicePlugin
#else
#define MOCKTESTMACRO
#endif
namespace MultiDevicePlugin {
struct AutoLoadContext {
    std::atomic<bool> isEnabled = {false};
    std::atomic<bool> isAlready = {false};
    std::atomic<bool> isLoadSuccess = {false};
    std::future<void> future;
    std::promise<void> promise;
    InferenceEngine::SoExecutableNetworkInternal executableNetwork;
    DeviceInformation  deviceInfo;
    std::vector<DeviceInformation> metaDevices;
    std::string networkPrecision;
    std::string errMessage;
    InferenceEngine::Task task;
    //ACTUALDEVICE's workName is same as it's deviceName,
    //CPU_HELP's workName is "CPU_HELP", and  deviceName is "CPU"
    //add workName is because of ACTUALDEVICE and CPU maybe all CPU,
    //they can't use the same workerQueue
    std::string workName = "";
};

enum AutoLoadContextIndex {
     CPU = 0,
     ACTUALDEVICE = 1,
     CONTEXTNUM = 2
};
class AutoSchedule : public MultiSchedule {
    public:
        using Ptr = std::shared_ptr<AutoSchedule>;
        void init(const Context::Ptr& context) override;
        IInferPtr CreateInferRequest() override;
        IInferPtr CreateInferRequestImpl(
                InferenceEngine::InputsDataMap networkInputs,
                InferenceEngine::OutputsDataMap networkOutputs) override;
        IInferPtr CreateInferRequestImpl(
                const std::vector<std::shared_ptr<const ov::Node>>& inputs,
                const std::vector<std::shared_ptr<const ov::Node>>& outputs) override;
        void WaitActualNetworkReady() const;
        virtual ~AutoSchedule();

    public:
        AutoLoadContext                           _loadContext[CONTEXTNUM];

    protected:
        void GenerateWorkers(const std::string& device,
                const InferenceEngine::SoExecutableNetworkInternal&
                executableNetwork) override;
        void ScheduleToWorkerInferRequest(InferenceEngine::Task,
                DeviceName preferred_device = "") override;

    private:
        void WaitFirstNetworkReady();
        void TryToLoadNetWork(AutoLoadContext& context,
                const std::string& modelPath,
                const InferenceEngine::CNNNetwork& network);

    private:
        InferenceEngine::IStreamsExecutor::Ptr   _executor;
        mutable std::once_flag                   _oc;
        std::once_flag                           _firstLoadOC;
        std::future<void>                        _firstLoadFuture;
        std::promise<void>                       _firstLoadPromise;
        bool                                     _exitFlag = {false};
        int                                      _cpuHelpInferCount = 0;
        std::atomic_size_t                       _numRequestsCreated = {0};
        AutoContext::Ptr                         _autoContext;
};

}  // namespace MultiDevicePlugin
