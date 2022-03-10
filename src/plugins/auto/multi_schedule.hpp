// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///////////////////////////////////////////////////////////////////////////////////////////////////
#pragma once

#include "base_schedule.hpp"

#ifdef  MULTIUNITTEST
#define MOCKTESTMACRO virtual
#define MultiDevicePlugin MockMultiDevicePlugin
#else
#define MOCKTESTMACRO
#endif

namespace MultiDevicePlugin {
struct ThisRequestExecutor : public IE::ITaskExecutor {
    explicit ThisRequestExecutor(WorkerInferRequest** ptr): _workptrptr{ptr} {}
    void run(IE::Task task) override {
        (*_workptrptr)->_task = std::move(task);
        (*_workptrptr)->_inferRequest->StartAsync();
    };
    WorkerInferRequest** _workptrptr = nullptr;
};

class MultiSchedule : public Schedule, public IE::ITaskExecutor {
public:
    using Ptr = std::shared_ptr<MultiSchedule>;
    using NotBusyWorkerRequests =
        IE::ThreadSafeBoundedPriorityQueue<std::pair<int, WorkerInferRequest*>>;
    IInferPtr CreateInferRequest() override;
    IInferPtr CreateInferRequestImpl(
        IE::InputsDataMap networkInputs,
        IE::OutputsDataMap networkOutputs) override;
    IE::IInferRequestInternal::Ptr CreateInferRequestImpl(
        const std::vector<std::shared_ptr<const ov::Node>>& inputs,
        const std::vector<std::shared_ptr<const ov::Node>>& outputs) override;
    void run(IE::Task inferTask) override;
    void init(const ScheduleContext::Ptr& sContext) override;
    Pipeline GetPipeline(const IInferPtr& syncRequestImpl,
        WorkerInferRequest** WorkerInferRequest) override;
    virtual ~MultiSchedule();

public:
    static thread_local WorkerInferRequest* _thisWorkerInferRequest;
    // have to use the const char* ptr rather than std::string due to a bug in old gcc versions,
    // the bug is e.g. manifesting on the old CentOS (and it's 4.8.x gcc) used in our testing
    // https://gcc.gnu.org/bugzilla/show_bug.cgi?id=81880
    static thread_local const char* _thisPreferredDeviceName;

protected:
    virtual void GenerateWorkers(const std::string& device,
        const IE::SoExecutableNetworkInternal& executableNetwork);
    static bool RunPipelineTask(IE::Task& inferPipelineTask,
        NotBusyWorkerRequests& idleWorkerRequests,
        const DeviceName& preferred_device);
    virtual void ScheduleToWorkerInferRequest(IE::Task,
        DeviceName preferred_device = "");

protected:
    IE::ThreadSafeQueue<IE::Task>  _inferPipelineTasks;
    DeviceMap<std::unique_ptr<IE::ThreadSafeQueue<IE::Task>>>
    _inferPipelineTasksDeviceSpecific;
    DeviceMap<NotBusyWorkerRequests> _idleWorkerRequests;
    DeviceMap<std::vector<WorkerInferRequest>> _workerRequests;
    mutable std::mutex _mutex;
    std::atomic_size_t  _numRequestsCreated = {0};
    MultiScheduleContext::Ptr _multiSContext;
};

struct IdleGuard {
    explicit IdleGuard(WorkerInferRequest* workerInferRequestPtr,
        MultiSchedule::NotBusyWorkerRequests& notBusyWorkerRequests) :
        _workerInferRequestPtr{workerInferRequestPtr},
        _notBusyWorkerRequests{&notBusyWorkerRequests} {
    }
    ~IdleGuard() {
        if (nullptr != _notBusyWorkerRequests) {
            _notBusyWorkerRequests->
            try_push(std::make_pair(_workerInferRequestPtr->_index,
                    _workerInferRequestPtr));
        }
    }
    MultiSchedule::NotBusyWorkerRequests* Release() {
        auto notBusyWorkerRequests = _notBusyWorkerRequests;
        _notBusyWorkerRequests = nullptr;
        return notBusyWorkerRequests;
    }
    WorkerInferRequest* _workerInferRequestPtr = nullptr;
    MultiSchedule::NotBusyWorkerRequests*  _notBusyWorkerRequests = nullptr;
};
}  // namespace MultiDevicePlugin
