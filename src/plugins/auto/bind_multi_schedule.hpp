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
class BinderMultiSchedule : public MultiSchedule {
public:
    using Ptr = std::shared_ptr<BinderMultiSchedule>;
    IInferPtr CreateInferRequest() override;
    IInferPtr CreateInferRequestImpl(IE::InputsDataMap networkInputs, IE::OutputsDataMap networkOutputs) override;
    IE::IInferRequestInternal::Ptr CreateInferRequestImpl(const std::vector<std::shared_ptr<const ov::Node>>& inputs,
                                                          const std::vector<std::shared_ptr<const ov::Node>>& outputs) override;
    void run(IE::Task inferTask) override;
    void init(const ScheduleContext::Ptr& sContext) override;
    Pipeline GetPipeline(const IInferPtr& syncRequestImpl, WorkerInferRequest** WorkerInferRequest) override;
    virtual ~BinderMultiSchedule();

public:
    static thread_local WorkerInferRequest*                         _thisWorkerInferRequest;
    static thread_local SoInfer                                     _sharedRequest;

protected:
    void GenerateWorkers(const std::string& device, const IE::SoExecutableNetworkInternal& executableNetwork) override;
    static bool RunPipelineTask(IE::Task& inferPipelineTask, NotBusyWorkerRequests& idleWorkerRequests, const DeviceName& preferred_device);
    bool ScheduleToWorkerInferRequest(IE::Task, DeviceName preferred_device = "") override;

protected:
    IE::ThreadSafeQueue<IE::Task>                             _inferPipelineTasks;
    DeviceMap<std::unique_ptr<IE::ThreadSafeQueue<IE::Task>>> _inferPipelineTasksDeviceSpecific;
    DeviceMap<NotBusyWorkerRequests>                          _idleWorkerRequests;
    DeviceMap<std::vector<WorkerInferRequest>>                _workerRequests;
    mutable std::mutex                                        _mutex;
    std::atomic_size_t                                        _numRequestsCreated = {0};
    MultiScheduleContext::Ptr                                 _bindMultiSContext;
};
}  // namespace MultiDevicePlugin
