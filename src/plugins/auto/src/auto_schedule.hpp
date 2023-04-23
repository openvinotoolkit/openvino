// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///////////////////////////////////////////////////////////////////////////////////////////////////
#pragma once

#include "schedule.hpp"

#ifdef  MULTIUNITTEST
#define MOCKTESTMACRO virtual
#define MultiDevicePlugin MockMultiDevicePlugin
#else
#define MOCKTESTMACRO
#endif
namespace MultiDevicePlugin {
struct ThisRequestExecutor : public IE::ITaskExecutor {
    explicit ThisRequestExecutor(WorkerInferRequest** ptr, MultiImmediateExecutor::Ptr executor = nullptr): _workptrptr{ptr}, _fallbackExec(executor) {}
    void run(IE::Task task) override {
        (*_workptrptr)->_task = std::move(task);
        (*_workptrptr)->_fallbackExec = _fallbackExec;
        (*_workptrptr)->_inferRequest->StartAsync();
    };
    WorkerInferRequest** _workptrptr = nullptr;
    MultiImmediateExecutor::Ptr _fallbackExec;
};
struct AutoLoadContext {
    std::atomic<bool> isEnabled = {false};
    std::atomic<bool> isAlready = {false};
    std::atomic<bool> isLoadSuccess = {false};
    std::atomic<bool> isReloadSuccess = {false};
    std::future<void> future;
    std::promise<void> promise;
    SoExecNetwork executableNetwork;
    DeviceInformation  deviceInfo;
    std::vector<DeviceInformation> metaDevices;
    std::string networkPrecision;
    std::string errMessage;
    IE::Task task;
    //ACTUALDEVICE's workName is same as it's deviceName,
    //CPU_HELP's workName is "CPU_HELP", and  deviceName is "CPU"
    //add workName is because of ACTUALDEVICE and CPU maybe all CPU,
    //they can't use the same workerQueue
    std::string workName = "";
};

enum AutoLoadContextIndex {
    CPU = 0,
    ACTUALDEVICE = 1,
    FALLBACKDEVICE = 2,
    CONTEXTNUM = 3
};
class AutoSchedule : public Schedule, public IE::ITaskExecutor {
public:
    using Ptr = std::shared_ptr<AutoSchedule>;
    void init(const ScheduleContext::Ptr& sContext) override;
    IInferPtr CreateInferRequest() override;
    IInferPtr CreateInferRequestImpl(IE::InputsDataMap networkInputs, IE::OutputsDataMap networkOutputs) override;
    IInferPtr CreateInferRequestImpl(const std::vector<std::shared_ptr<const ov::Node>>& inputs,
                                                          const std::vector<std::shared_ptr<const ov::Node>>& outputs) override;
    void run(IE::Task inferTask) override;
    Pipeline GetPipeline(const IInferPtr& syncRequestImpl, WorkerInferRequest** WorkerInferRequest) override;
    void WaitActualNetworkReady() const;
    virtual ~AutoSchedule();

public:
    static thread_local WorkerInferRequest* _thisWorkerInferRequest;
    // have to use the const char* ptr rather than std::string due to a bug in old gcc versions,
    // the bug is e.g. manifesting on the old CentOS (and it's 4.8.x gcc) used in our testing
    // https://gcc.gnu.org/bugzilla/show_bug.cgi?id=81880
    static thread_local const char* _thisPreferredDeviceName;
    AutoLoadContext                           _loadContext[CONTEXTNUM];
    std::unique_ptr<AutoLoadContext[]>        _pCTPUTLoadContext = nullptr;
    size_t                                    _nCTputDeviceNums = 0;

protected:
    void GenerateWorkers(const std::string& device, const SoExecNetwork& executableNetwork);
    bool ScheduleToWorkerInferRequest(IE::Task, DeviceName preferred_device = "");
    static bool RunPipelineTask(IE::Task& inferPipelineTask, NotBusyPriorityWorkerRequests& idleWorkerRequests,
                                const DeviceName& preferred_device);
    std::string GetLogTag() const noexcept;
    DeviceMap<NotBusyPriorityWorkerRequests>                _idleWorkerRequests;
    AutoScheduleContext::Ptr                                _autoSContext;
    std::atomic_size_t                                      _numRequestsCreated = {0};
    DeviceMap<std::vector<WorkerInferRequest>>              _workerRequests;

private:
    /**
     * @brief wait for one of the executable network to finish loading.
     * @return An SoPtr object hold an available executable network loaded to HW device.
     * @note An exception will be thrown if all loading of network to hw device fails.
     */
    SoExecNetwork WaitFirstNetworkReady();
    void TryToLoadNetWork(AutoLoadContext& context, const std::string& modelPath, const IE::CNNNetwork& network, bool isCumulative);
    bool selectOtherDevice(const std::string& currentDeviceName);
    IE::Task releaseActualdeviceTask;

private:
    IE::ThreadSafeQueue<IE::Task>                             _inferPipelineTasks;
    DeviceMap<std::unique_ptr<IE::ThreadSafeQueue<IE::Task>>> _inferPipelineTasksDeviceSpecific;
    SoExecNetwork                                             _passthroughExeNet;
    Time                                                      _cpuHelpReleaseTime;
    size_t                                                    _cpuHelpInferCount = 0;
    double                                                    _cpuHelpFps = 0.0;
    std::string                                               _LogTag;
    IE::IStreamsExecutor::Ptr                                 _executor;
    mutable std::once_flag                                    _oc;
    std::once_flag                                            _firstLoadOC;
    std::future<void>                                         _firstLoadFuture;
    std::promise<void>                                        _firstLoadPromise;
    bool                                                      _exitFlag = {false};
};

}  // namespace MultiDevicePlugin
