// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///////////////////////////////////////////////////////////////////////////////////////////////////
#pragma once

#include "base_schedule.hpp"
#include "threading/ie_thread_safe_containers.hpp"

#ifdef  MULTIUNITTEST
#define MOCKTESTMACRO virtual
#define MultiDevicePlugin MockMultiDevicePlugin
#else
#define MOCKTESTMACRO
#endif

namespace MultiDevicePlugin {
struct ThisRequestExecutor : public InferenceEngine::ITaskExecutor {
    explicit ThisRequestExecutor(WorkerInferRequest** ptr):_workptrptr{ptr} {}
    void run(InferenceEngine::Task task) override {
        (*_workptrptr)->_task = std::move(task);
        (*_workptrptr)->_inferRequest->StartAsync();
    };
    WorkerInferRequest** _workptrptr = nullptr;
};

class MultiSchedule : public Schedule, public InferenceEngine::ITaskExecutor {
    public:
        using Ptr = std::shared_ptr<MultiSchedule>;
        using NotBusyWorkerRequests =
            InferenceEngine::ThreadSafeBoundedPriorityQueue<std::pair<int, WorkerInferRequest*>>;
        InferenceEngine::IInferRequestInternal::Ptr CreateInferRequest() override;
        InferenceEngine::IInferRequestInternal::Ptr CreateInferRequestImpl(
                InferenceEngine::InputsDataMap networkInputs,
                InferenceEngine::OutputsDataMap networkOutputs) override;
        InferenceEngine::IInferRequestInternal::Ptr CreateInferRequestImpl(
                const std::vector<std::shared_ptr<const ov::Node>>& inputs,
                const std::vector<std::shared_ptr<const ov::Node>>& outputs) override;
        void run(InferenceEngine::Task inferTask) override;
        void init(const Context::Ptr& context) override;
        Pipeline GetPipeline(const InferenceEngine::IInferRequestInternal::Ptr& syncRequestImpl,
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
                const InferenceEngine::SoExecutableNetworkInternal& executableNetwork);
        static bool RunPipelineTask(InferenceEngine::Task& inferPipelineTask,
                NotBusyWorkerRequests& idleWorkerRequests,
                const DeviceName& preferred_device);
        virtual void ScheduleToWorkerInferRequest(InferenceEngine::Task,
                DeviceName preferred_device = "");

    protected:
        InferenceEngine::ThreadSafeQueue<InferenceEngine::Task>                      _inferPipelineTasks;
        DeviceMap<std::unique_ptr<InferenceEngine::ThreadSafeQueue<InferenceEngine::Task>>> _inferPipelineTasksDeviceSpecific;
        DeviceMap<NotBusyWorkerRequests>                            _idleWorkerRequests;
        DeviceMap<std::vector<WorkerInferRequest>>                  _workerRequests;
        mutable std::mutex                                          _mutex;
        std::atomic_size_t                                          _numRequestsCreated = {0};
        MultiContext::Ptr                                           _multiContext;
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
    WorkerInferRequest*     _workerInferRequestPtr = nullptr;
    MultiSchedule::NotBusyWorkerRequests*  _notBusyWorkerRequests = nullptr;
};
}  // namespace MultiDevicePlugin
