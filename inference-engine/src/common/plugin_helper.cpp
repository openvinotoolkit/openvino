#include "plugin_helper.hpp"

namespace PluginHelper {

using namespace InferenceEngine;

void CreateWorkers(SoExecutableNetworkInternal&                       executableNetwork,
                   DeviceMap<std::vector<WorkerInferRequest>>&        workerRequestsMap,
                   DeviceMap<NotBusyWorkerRequests>&                  idleWorkerRequestsMap,
                   ThreadSafeQueue<Task>&                             inferPipelineTasks,
                   DeviceMap<std::unique_ptr<ThreadSafeQueue<Task>>>& inferPipelineTasksDeviceSpecific,
                   uint32_t                                           optimalNum,
                   const std::string&                                 device,
                   std::function<void(Task, const DeviceName&)>       scheduleFunc) {
    auto& workerRequests = workerRequestsMap[device];
    workerRequests.resize(optimalNum);
    inferPipelineTasksDeviceSpecific[device] = std::unique_ptr<ThreadSafeQueue<Task>>(new ThreadSafeQueue<Task>);
    auto& idleWorkerRequests = idleWorkerRequestsMap[device];
    idleWorkerRequests.set_capacity(optimalNum);
    auto* idleWorkerRequestsPtr = &(idleWorkerRequests);
    for (auto&& workerRequest : workerRequests) {
        workerRequest._inferRequest = { executableNetwork, executableNetwork->CreateInferRequest() };
        auto* workerRequestPtr = &workerRequest;
        IE_ASSERT(idleWorkerRequests.try_push(workerRequestPtr));
        workerRequest._inferRequest->SetCallback(
            [workerRequestPtr, device, idleWorkerRequestsPtr, scheduleFunc, &inferPipelineTasks, &inferPipelineTasksDeviceSpecific] (std::exception_ptr exceptionPtr) mutable {
                PluginHelper::IdleGuard idleGuard{workerRequestPtr, *idleWorkerRequestsPtr};
                workerRequestPtr->_exceptionPtr = exceptionPtr;
                {
                    // this is the last task in pipeline
                    auto capturedTask = std::move(workerRequestPtr->_task);
                    capturedTask();
                }
                // try to return the request to the idle list (fails if the overall object destruction has began)
                if (idleGuard.Release()->try_push(workerRequestPtr)) {
                    // let's try to pop a task, as we know there is at least one idle request, schedule if succeeded
                    // if no device-agnostic tasks, let's try pop the device specific task, schedule if succeeded
                    Task t;
                    if (inferPipelineTasks.try_pop(t))
                        scheduleFunc(std::move(t), "");
                    else if (inferPipelineTasksDeviceSpecific[device]->try_pop(t))
                        scheduleFunc(std::move(t), device);
                }
            });
  }
}
}