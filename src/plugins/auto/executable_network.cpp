// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///////////////////////////////////////////////////////////////////////////////////////////////////
#include <mutex>
#include <string>
#include <vector>
#include <memory>
#include <utility>
#include <map>
#include <unordered_map>

#include "ie_icore.hpp"
#include "ie_metric_helpers.hpp"
#include <ie_plugin_config.hpp>
#include "executable_network.hpp"
#include "async_infer_request.hpp"
#include "plugin.hpp"

#include "ngraph/opsets/opset1.hpp"
#include "transformations/utils/utils.hpp"
#include "utils/log_util.hpp"

#include "itt.hpp"
// ------------------------------MultiDeviceExecutableNetwork----------------------------
namespace MultiDevicePlugin {
using namespace InferenceEngine;

namespace {
std::string GetNetworkPrecision(const InferenceEngine::CNNNetwork &network) {
    auto nGraphFunc = network.getFunction();
    bool isINTModel = ngraph::op::util::has_op_with_type<ngraph::op::FakeQuantize>(nGraphFunc);
    if (isINTModel) {
        return METRIC_VALUE(INT8);
    }
    for (auto & node : nGraphFunc->get_ordered_ops()) {
        if (std::dynamic_pointer_cast<ngraph::opset1::Convolution>(node) ||
            std::dynamic_pointer_cast<ngraph::opset1::GroupConvolution>(node) ||
            std::dynamic_pointer_cast<ngraph::opset1::GroupConvolutionBackpropData>(node) ||
            std::dynamic_pointer_cast<ngraph::opset1::ConvolutionBackpropData>(node)) {
            auto layerType = node->input(1).get_element_type().get_type_name();
            if (layerType == "f32")
                return METRIC_VALUE(FP32);
            if (layerType == "f16")
                return METRIC_VALUE(FP16);
        }
    }
    return METRIC_VALUE(FP32);
}
}  // namespace

thread_local MultiDeviceExecutableNetwork::WorkerInferRequest* MultiDeviceExecutableNetwork::_thisWorkerInferRequest = nullptr;
// TODO: revert to the plain variable (see header file), when we moved to the next CentOS 8.x in our support matrix
thread_local const char* MultiDeviceExecutableNetwork::_thisPreferredDeviceName = "";

struct IdleGuard {
    explicit IdleGuard(MultiDeviceExecutableNetwork::WorkerInferRequest* workerInferRequestPtr,
                       MultiDeviceExecutableNetwork::NotBusyWorkerRequests& notBusyWorkerRequests) :
        _workerInferRequestPtr{workerInferRequestPtr},
        _notBusyWorkerRequests{&notBusyWorkerRequests} {
    }
    ~IdleGuard() {
        if (nullptr != _notBusyWorkerRequests) {
            _notBusyWorkerRequests->try_push(std::make_pair(_workerInferRequestPtr->_index, _workerInferRequestPtr));
        }
    }
    MultiDeviceExecutableNetwork::NotBusyWorkerRequests* Release() {
        auto notBusyWorkerRequests = _notBusyWorkerRequests;
        _notBusyWorkerRequests = nullptr;
        return notBusyWorkerRequests;
    }
    MultiDeviceExecutableNetwork::WorkerInferRequest*     _workerInferRequestPtr = nullptr;
    MultiDeviceExecutableNetwork::NotBusyWorkerRequests*  _notBusyWorkerRequests = nullptr;
};

MultiDeviceExecutableNetwork::MultiDeviceExecutableNetwork(const DeviceMap<InferenceEngine::SoExecutableNetworkInternal>&       networksPerDevice,
                                                           const std::vector<DeviceInformation>&                                networkDevices,
                                                           const std::unordered_map<std::string, InferenceEngine::Parameter>&   config,
                                                           const bool                                                           needPerfCounters) :
    InferenceEngine::ExecutableNetworkThreadSafeDefault(nullptr, std::make_shared<InferenceEngine::ImmediateExecutor>()),
    _devicePriorities{networkDevices},
    _devicePrioritiesInitial{networkDevices},
    _networksPerDevice{networksPerDevice},
    _config{config},
    _needPerfCounters{needPerfCounters} {
    _taskExecutor.reset();
    for (auto&& networkValue : _networksPerDevice) {
        auto& device  = networkValue.first;
        auto& network = networkValue.second;
        GenerateWorkers(device, network);
    }
}

void MultiDeviceExecutableNetwork::GenerateWorkers(const std::string& device, const SoExecutableNetworkInternal& executableNetwork) {
    std::string realDeviceName;
    if (device == "CPU_HELP") {
        realDeviceName = "CPU";
    } else {
        realDeviceName = device;
    }
    auto itNumRequests = std::find_if(_devicePriorities.cbegin(), _devicePriorities.cend(),
                                      [&realDeviceName](const DeviceInformation& d){ return d.deviceName == realDeviceName;});
    unsigned int optimalNum = 0;
    try {
        optimalNum = executableNetwork->GetMetric(METRIC_KEY(OPTIMAL_NUMBER_OF_INFER_REQUESTS)).as<unsigned int>();
    } catch (const InferenceEngine::Exception &iie) {
        IE_THROW()
            << "Every device used with the Multi-Device should "
            << "support OPTIMAL_NUMBER_OF_INFER_REQUESTS ExecutableNetwork metric. "
            << "Failed to query the metric for the " << device << " with error:" << iie.what();
    }
    const auto numRequests = (_devicePriorities.end() == itNumRequests ||
                              itNumRequests->numRequestsPerDevices == -1) ? optimalNum : itNumRequests->numRequestsPerDevices;
    auto& workerRequests = _workerRequests[device];
    auto& idleWorkerRequests = _idleWorkerRequests[device];
    workerRequests.resize(numRequests);
    _inferPipelineTasksDeviceSpecific[device] = std::unique_ptr<ThreadSafeQueue<Task>>(new ThreadSafeQueue<Task>);
    auto* idleWorkerRequestsPtr = &(idleWorkerRequests);
    idleWorkerRequests.set_capacity(numRequests);
    int num = 0;
    for (auto&& workerRequest : workerRequests) {
        workerRequest._inferRequest = {executableNetwork->CreateInferRequest(), executableNetwork._so};
        auto* workerRequestPtr = &workerRequest;
        workerRequestPtr->_index = num++;
        IE_ASSERT(idleWorkerRequests.try_push(std::make_pair(workerRequestPtr->_index, workerRequestPtr)) == true);
        workerRequest._inferRequest->SetCallback(
            [workerRequestPtr, this, device, idleWorkerRequestsPtr] (std::exception_ptr exceptionPtr) mutable {
                IdleGuard idleGuard{workerRequestPtr, *idleWorkerRequestsPtr};
                workerRequestPtr->_exceptionPtr = exceptionPtr;
                {
                    auto capturedTask = std::move(workerRequestPtr->_task);
                    capturedTask();
                }
                // try to return the request to the idle list (fails if the overall object destruction has began)
                if (idleGuard.Release()->try_push(std::make_pair(workerRequestPtr->_index, workerRequestPtr))) {
                    // let's try to pop a task, as we know there is at least one idle request, schedule if succeeded
                    // if no device-agnostic tasks, let's try pop the device specific task, schedule if succeeded
                    Task t;
                    if (_inferPipelineTasks.try_pop(t))
                        ScheduleToWorkerInferRequest(std::move(t));
                    else if (_inferPipelineTasksDeviceSpecific[device]->try_pop(t))
                        ScheduleToWorkerInferRequest(std::move(t), device);
                }
            });
    }
}

MultiDeviceExecutableNetwork::MultiDeviceExecutableNetwork(const std::string&                         modelPath,
                                                           const InferenceEngine::CNNNetwork&         network,
                                                           const std::vector<DeviceInformation>&      metaDevices,
                                                           const std::string&                         strDevices,
                                                           MultiDeviceInferencePlugin*                plugin,
                                                           const AutoContext&                         context,
                                                           const bool                                 needPerfCounters)
                                                           : _devicePriorities{metaDevices}
                                                           , _devicePrioritiesInitial{metaDevices}
                                                           , _needPerfCounters(needPerfCounters)
                                                           , _multiPlugin(plugin)
                                                           , _context(context)
                                                           , _workModeIsAUTO(true)
                                                           , _network(network) {
    if (_multiPlugin->GetCore() == nullptr) {
        IE_THROW() << "Please, work with " << _multiPlugin->GetName() << " device via InferencEngine::Core object";
    }

    if (modelPath.empty() && network.getFunction() == nullptr) {
        IE_THROW() << "MULTI " << _multiPlugin->GetName() << " device supports just ngraph network representation";
    }

    _core = _multiPlugin->GetCore(); // shared_ptr that holds the Core
    _config[MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES] = strDevices;

    std::string profilingTask = "MultiDeviceExecutableNetwork::MultiDeviceExecutableNetwork:AutoMode";

    // loadContext[ACTUALDEVICE] is always enabled,
    // when there is CPU and there are more than two devices, loadContext[CPU] is enabled
    _loadContext[ACTUALDEVICE].isEnabled = true;
    _loadContext[ACTUALDEVICE].networkPrecision = GetNetworkPrecision(network);
    _loadContext[ACTUALDEVICE].metaDevices = metaDevices;
    _loadContext[ACTUALDEVICE].deviceInfo = _multiPlugin->SelectDevice(metaDevices,
            _loadContext[ACTUALDEVICE].networkPrecision, _context.modelPriority);
    LOG_INFO("[AUTOPLUGIN]:select device:%s", _loadContext[ACTUALDEVICE].deviceInfo.deviceName.c_str());
    bool isActualDevCPU =
        _loadContext[ACTUALDEVICE].deviceInfo.deviceName.find("CPU") != std::string::npos;
    // if Actual device is CPU, disabled _loadContext[CPU], only use _loadContext[ACTUALDEVICE]
    if (isActualDevCPU) {
        _loadContext[CPU].isEnabled = false;
    } else {
        const auto CPUIter = std::find_if(metaDevices.begin(), metaDevices.end(),
                [=](const DeviceInformation& d)->bool{return d.deviceName.find("CPU") != std::string::npos;});
        // if have CPU Device,  enable _loadContext[CPU]
        if (CPUIter != metaDevices.end()) {
            _loadContext[CPU].isEnabled = true;
            _loadContext[CPU].deviceInfo = *CPUIter;
            _loadContext[CPU].deviceInfo.config[CONFIG_KEY(PERFORMANCE_HINT)] =
                InferenceEngine::PluginConfigParams::LATENCY;
            _loadContext[CPU].workName = "CPU_HELP";
            LOG_INFO("[AUTOPLUGIN]:will load CPU for accelerator");
        } else {
            _loadContext[CPU].isEnabled = false;
        }
    }

    // initialize the rest members of load context
    for (int i = 0; i < CONTEXTNUM; i++) {
         if (_loadContext[i].isEnabled) {
             _loadContext[i].future =  _loadContext[i].promise.get_future();
              auto* contextPtr = &_loadContext[i];
             _loadContext[i].task = [this, contextPtr, modelPath, network]() mutable {
                      TryToLoadNetWork(*contextPtr, modelPath, network);
                      if (contextPtr->isLoadSuccess) {
                          if (contextPtr->workName.empty()) {
                                contextPtr->workName = contextPtr->deviceInfo.deviceName;
                          }
                          GenerateWorkers(contextPtr->workName, contextPtr->executableNetwork);
                          //need lock
                          {
                             std::lock_guard<std::mutex> lock(_confMutex);
                             _config.insert(contextPtr->deviceInfo.config.begin(),
                                            contextPtr->deviceInfo.config.end());
                          }
                          contextPtr->isAlready = true;
                          auto& deviceName = contextPtr->deviceInfo.deviceName;
                          LOG_INFO("[AUTOPLUGIN]:device:%s loading Network finished",
                                  deviceName.c_str());
                          std::vector<std::string> supported_config_keys =
                              _core->GetMetric(deviceName, METRIC_KEY(SUPPORTED_CONFIG_KEYS));
                          // there is log mutex in LOG_DEBUG, add _configMutex just want to print them all together
                          // toDo maybe neet to implement LOG_RUN(task, LOG_LEVEL) to run some debug code.
                          std::lock_guard<std::mutex> lock(_confMutex);
                          for (const auto& cfg : supported_config_keys) {
                              try {
                                  LOG_DEBUG("[AUTOPLUGIN]:device:%s, GetConfig:%s=%s", deviceName.c_str(),
                                          cfg.c_str(), contextPtr->executableNetwork->GetConfig(cfg).as<std::string>().c_str());
                              } catch (...) {
                              }
                          }
                      }
                      contextPtr->promise.set_value();
                      // the first load network process finished
                      std::call_once(_firstLoadOC, [this] () {
                              _firstLoadPromise.set_value();
                              });
             };
         }
    }

    OV_ITT_SCOPED_TASK(itt::domains::MULTIPlugin, openvino::itt::handle(profilingTask));
    if (_loadContext[CPU].isEnabled) {
        _firstLoadFuture = _firstLoadPromise.get_future();
        // will not wait for loading accelerator network,
        // so the executor can't be destroyed before finished the task,
        // so use executor as a member of MultiDeviceExecutableNetwork.
        _executor = InferenceEngine::ExecutorManager::getInstance()->getIdleCPUStreamsExecutor(
                IStreamsExecutor::Config{"AutoDeviceAsyncLoad",
                static_cast<int>(std::thread::hardware_concurrency()) /* max possible #streams*/,
                0 /*default threads per stream, workaround for ticket 62376*/,
                IStreamsExecutor::ThreadBindingType::NONE});
        for (auto&& device : metaDevices) {
            // initialize containers before run async task
            _idleWorkerRequests[device.deviceName];
            _workerRequests[device.deviceName];
            _inferPipelineTasksDeviceSpecific[device.deviceName] = nullptr;
        }
        _idleWorkerRequests["CPU_HELP"];
        _workerRequests["CPU_HELP"];
        _inferPipelineTasksDeviceSpecific["CPU_HELP"] = nullptr;
        _executor->run(_loadContext[CPU].task);
        _executor->run(_loadContext[ACTUALDEVICE].task);
        auto recycleTask = [this]() mutable {
            WaitActualNetworkReady();
            while (!_exitFlag && _loadContext[ACTUALDEVICE].isAlready) {
                // handle the case of ACTUAL faster than CPU
                _loadContext[CPU].future.wait();
                // clean up helper infer requests
                // first, wait for all the remaining requests to finish
                for (auto& iter : _workerRequests["CPU_HELP"]) {
                    iter._inferRequest._ptr->Wait(InferRequest::WaitMode::RESULT_READY);
                }
                // late enough to check the idle queue now
                // second, check the idle queue if all requests are in place
                size_t destroynum = 0;
                std::pair<int, WorkerInferRequest*> worker;
                while (_idleWorkerRequests["CPU_HELP"].try_pop(worker))
                    destroynum++;
                if (destroynum == _workerRequests["CPU_HELP"].size()) {
                    std::lock_guard<std::mutex> lock(_confMutex);
                    _workerRequests["CPU_HELP"].clear();
                    _loadContext[CPU].executableNetwork._ptr.reset();
                    _loadContext[CPU].executableNetwork._so.reset();
                    break;
                }
            }
        };
        _executor->run(std::move(recycleTask));
    } else {
        // only one device need to load network, do not need to load it async
        _loadContext[ACTUALDEVICE].task();
    }
    WaitFirstNetworkReady();
}
void MultiDeviceExecutableNetwork::TryToLoadNetWork(AutoLoadContext& context,
                                                    const std::string& modelPath,
                                                    const InferenceEngine::CNNNetwork& network) {
    auto& device = context.deviceInfo.deviceName;
    auto& deviceConfig = context.deviceInfo.config;
    auto& deviceList = context.metaDevices;
    bool curDevIsCPU = (device.find("CPU") != std::string::npos);
    try {
        if (!modelPath.empty()) {
            context.executableNetwork = _core->LoadNetwork(modelPath, device, deviceConfig);
        } else {
            context.executableNetwork = _core->LoadNetwork(network, device, deviceConfig);
        }
        context.isLoadSuccess = true;
    } catch (const std::exception& e) {
        context.errMessage += device + ":" + e.what();
        context.isLoadSuccess = false;
    }

    if (context.isLoadSuccess || curDevIsCPU) {
        return;
    }

    // need to reload network, unregister it's priority
    // there maybe potential issue.
    // for example they are dGPU, VPUX, iGPU, customer want to LoadNetwork with
    // configure 0 dGPU, 1 VPUX, if dGPU load failed,
    // the result will be not sure, maybe two network are loaded into VPUX,
    // maybe 0 is loaded to VPUX, 1 is loaded to iGPU
    _multiPlugin->UnregisterPriority(_context.modelPriority, context.deviceInfo.uniqueName);
    // remove the current device from deviceList
    auto eraseDevice = std::find_if(deviceList.begin(), deviceList.end(),
            [device](DeviceInformation& d){
            return d.deviceName == device;
            });
    deviceList.erase(eraseDevice);

    if (deviceList.empty()) {
        return;
    }

    // select next candidate device
    try {
        context.deviceInfo = _multiPlugin->SelectDevice(deviceList,
                context.networkPrecision, _context.modelPriority);
    }
    catch (const std::exception& e) {
        return;
    }

    // if the select device is CPU, need to check the config of _loadContext[CPU]
    // if they are same, do not need to load again
    curDevIsCPU = (context.deviceInfo.deviceName.find("CPU") != std::string::npos);
    if (curDevIsCPU) {
        auto compare = [](std::map<std::string, std::string>& a,
                std::map<std::string, std::string>& b) -> bool {
            if (a.size() != b.size()) {
                return false;
            }
            for (auto& item : a) {
                auto bIter = b.find(item.first);
                if (bIter != b.end()) {
                    if (bIter->second != item.second) {
                        return false;
                    }
                } else {
                    return false;
                }
            }
            return true;
        };
        if (compare(context.deviceInfo.config, _loadContext[CPU].deviceInfo.config)) {
            return;
        }
    }

    LOG_DEBUG("[AUTOPLUGIN] try to load %s", context.deviceInfo.deviceName.c_str());
    // try to load this candidate device
    TryToLoadNetWork(context, modelPath, network);
}

void MultiDeviceExecutableNetwork::WaitFirstNetworkReady() {
    if (_firstLoadFuture.valid()) {
        // wait for the first loading finished
        _firstLoadFuture.wait();
    }

    // check if there is any device that have loaded network successfully
    for (int i = CONTEXTNUM - 1; i >= 0; i--) {
        if (_loadContext[i].isEnabled && _loadContext[i].isAlready) {
            return;
        }
    }

    // the first loading is failed, wait for another loading
    for (int i = CONTEXTNUM - 1; i >= 0; i--) {
        if (_loadContext[i].isEnabled) {
            _loadContext[i].future.wait();
            // check if loading is successful
            if (_loadContext[i].isAlready) {
                return;
            }
        }
    }

    //print errMessage
    for (int i = CONTEXTNUM - 1; i >= 0; i--) {
        if (_loadContext[i].isEnabled) {
            LOG_ERROR("[AUTOPLUGIN] load failed, %s", _loadContext[i].errMessage.c_str());
        }
    }

    IE_THROW() << "[AUTOPLUGIN] load all devices failed";
}

void MultiDeviceExecutableNetwork::WaitActualNetworkReady() const {
    OV_ITT_SCOPED_TASK(itt::domains::MULTIPlugin, "MultiDeviceExecutableNetwork::WaitActualNetworkReady");
    // Maybe different API will call this function, so add call once here
    // for every MultiDeviceExecutableNetwork instance
    std::call_once(_oc, [this] () {
               if (_loadContext[ACTUALDEVICE].future.valid()) {
                   _loadContext[ACTUALDEVICE].future.wait();
               }
               });
}

void MultiDeviceExecutableNetwork::ScheduleToWorkerInferRequest(Task inferPipelineTask, DeviceName preferred_device) {
    std::vector<DeviceInformation> devices;
    // AUTO work mode
    if (_workModeIsAUTO) {
        if (!preferred_device.empty()) {
            // if the device needed by customer is not ready, need to wait for it
            WaitActualNetworkReady();
            // the preferred_device should be the selected device in AUTO work mode
            if (preferred_device != _loadContext[ACTUALDEVICE].deviceInfo.deviceName) {
                IE_THROW(NotFound) << "The preferred device should be the selected device";
            }
            devices.push_back(_loadContext[ACTUALDEVICE].deviceInfo);
        } else {
            // _acceleratorDevice could be the same as _cpuDevice, such as AUTO:CPU
            if (_loadContext[ACTUALDEVICE].isAlready) {
                devices.push_back(_loadContext[ACTUALDEVICE].deviceInfo);
            } else {
                // replace deviceName with workName, so schedule can select correct
                // idleWorkerQueue
                auto deviceInfo =  _loadContext[CPU].deviceInfo;
                deviceInfo.deviceName = _loadContext[CPU].workName;
                devices.push_back(std::move(deviceInfo));
            }
        }
    } else {
        devices = [&] {
            std::lock_guard<std::mutex> lock(_mutex);
            return _devicePriorities;
        }();
    }
    for (auto&& device : devices) {
        if (!preferred_device.empty() && (device.deviceName != preferred_device))
            continue;
        if (RunPipelineTask(inferPipelineTask, _idleWorkerRequests[device.deviceName], preferred_device)) {
            return;
        }
    }

    // no vacant requests this time, storing the task to the respective queue
    if (!preferred_device.empty())
        _inferPipelineTasksDeviceSpecific[preferred_device]->push(std::move(inferPipelineTask));
    else
        _inferPipelineTasks.push(std::move(inferPipelineTask));
}

bool MultiDeviceExecutableNetwork::RunPipelineTask(Task& inferPipelineTask,
                                            NotBusyWorkerRequests& idleWorkerRequests,
                                            const DeviceName& preferred_device) {
  WorkerInferRequest *workerRequestPtr = nullptr;
  std::pair<int, WorkerInferRequest*> worker;
  if (idleWorkerRequests.try_pop(worker)) {
      workerRequestPtr = worker.second;
      IdleGuard idleGuard{workerRequestPtr, idleWorkerRequests};
      _thisWorkerInferRequest = workerRequestPtr;
      {
          auto capturedTask = std::move(inferPipelineTask);
          capturedTask();
      }
      idleGuard.Release();
      return true;
  }
  return false;
}

void MultiDeviceExecutableNetwork::run(Task inferPipelineTask) {
    ScheduleToWorkerInferRequest(std::move(inferPipelineTask), _thisPreferredDeviceName);
}

MultiDeviceExecutableNetwork::~MultiDeviceExecutableNetwork() {
    if (_workModeIsAUTO) {
        // this is necessary to guarantee member destroyed after getting future
        if (_loadContext[CPU].isEnabled) {
            _exitFlag = true;
            _loadContext[CPU].future.wait();
            WaitActualNetworkReady();
            // it's necessary to wait the loading network threads to stop here.
            InferenceEngine::ExecutorManager::getInstance()->clear("AutoDeviceAsyncLoad");
            _executor.reset();
        }
        _multiPlugin->UnregisterPriority(_context.modelPriority,
                _loadContext[ACTUALDEVICE].deviceInfo.uniqueName);
    }
    {
        std::lock_guard<std::mutex> lock(_mutex);
        _devicePriorities.clear();
    }
    /* NOTE: The only threads that use `MultiDeviceExecutableNetwork` worker infer requests' threads.
     *       But AsyncInferRequest destructor should wait for all asynchronous tasks by the request
     */
    for (auto&& idleWorker : _idleWorkerRequests) {
        // stop accepting any idle requests back (for re-scheduling)
        idleWorker.second.set_capacity(0);
    }
    {
        std::lock_guard<std::mutex> lock(_confMutex);
        _workerRequests.clear();
    }
}

std::shared_ptr<InferenceEngine::RemoteContext> MultiDeviceExecutableNetwork::GetContext() const {
    if (_workModeIsAUTO) {
        WaitActualNetworkReady();
        return _loadContext[ACTUALDEVICE].executableNetwork->GetContext();
    }
    auto devices = [&] {
        std::lock_guard<std::mutex> lock(_mutex);
        return _devicePriorities;
    }();

    std::string devices_names;
    for (auto&& device : devices) {
        devices_names += device.deviceName + " ";
        const auto& n  = _networksPerDevice.at(device.deviceName);
        try {
            return n->GetContext();
        } catch (const NotImplemented&) {}
    }
    IE_THROW(NotImplemented) << "None of the devices in the MULTI device has an associated remote context."
                             << " Current list of devices allowed via the DEVICE_PRIORITIES config: " << devices_names;
}

std::shared_ptr<InferenceEngine::ICore> MultiDeviceExecutableNetwork::GetCore() const {
    return _plugin->GetCore();
}

InferenceEngine::IInferRequestInternal::Ptr MultiDeviceExecutableNetwork::CreateInferRequestImpl(
    const std::vector<std::shared_ptr<const ov::Node>>& inputs,
    const std::vector<std::shared_ptr<const ov::Node>>& outputs) {
    auto num = _numRequestsCreated++;
    size_t sum = 0;
    InferenceEngine::SoIInferRequestInternal request_to_share_blobs_with;

    if (_workModeIsAUTO) {
        if (!_loadContext[CPU].isEnabled && _loadContext[ACTUALDEVICE].isAlready) {
            auto& dev_requests = _workerRequests[_loadContext[ACTUALDEVICE].deviceInfo.deviceName];
            if (num < dev_requests.size()) {
                request_to_share_blobs_with = dev_requests.at(num)._inferRequest;
            }
        }
        // if user creates more infer request than the device optimal value, fall back to default memory
        return std::make_shared<MultiDeviceInferRequest>(inputs, outputs, request_to_share_blobs_with);
    }

    // borrowing device-specific blobs from the underlying requests for the device-agnostic, user-facing requests
    // this allows to potentially save on the data-copy later (if the requests are scheduled in the same order)
    for (const auto& device : _devicePrioritiesInitial) {
        auto& dev_requests = _workerRequests[device.deviceName];
        if ((num - sum) < dev_requests.size()) {
            request_to_share_blobs_with = dev_requests.at(num - sum)._inferRequest;
            break;
        }
        sum += dev_requests.size();
    }
    return std::make_shared<MultiDeviceInferRequest>(inputs, outputs, request_to_share_blobs_with);
}

InferenceEngine::IInferRequestInternal::Ptr MultiDeviceExecutableNetwork::CreateInferRequestImpl(InferenceEngine::InputsDataMap networkInputs,
                                                                                                InferenceEngine::OutputsDataMap networkOutputs) {
    auto num = _numRequestsCreated++;
    size_t sum = 0;
    InferenceEngine::SoIInferRequestInternal request_to_share_blobs_with;

    if (_workModeIsAUTO) {
        if (!_loadContext[CPU].isEnabled && _loadContext[ACTUALDEVICE].isAlready) {
            auto& dev_requests = _workerRequests[_loadContext[ACTUALDEVICE].deviceInfo.deviceName];
            if (num < dev_requests.size()) {
                request_to_share_blobs_with = dev_requests.at(num)._inferRequest;
            }
        }
        // if user creates more infer request than the device optimal value, fall back to default memory
        return std::make_shared<MultiDeviceInferRequest>(networkInputs, networkOutputs, request_to_share_blobs_with);
    }

    // borrowing device-specific blobs from the underlying requests for the device-agnostic, user-facing requests
    // this allows to potentially save on the data-copy later (if the requests are scheduled in the same order)
    for (const auto& device : _devicePrioritiesInitial) {
        auto& dev_requests = _workerRequests[device.deviceName];
        if ((num - sum) < dev_requests.size()) {
            request_to_share_blobs_with = dev_requests.at(num - sum)._inferRequest;
            break;
        }
        sum += dev_requests.size();
    }
    return std::make_shared<MultiDeviceInferRequest>(networkInputs, networkOutputs, request_to_share_blobs_with);
}

IInferRequestInternal::Ptr MultiDeviceExecutableNetwork::CreateInferRequest() {
    IInferRequestInternal::Ptr syncRequestImpl;
    if (this->_plugin && this->_plugin->GetCore() && GetCore()->isNewAPI())
        syncRequestImpl = CreateInferRequestImpl(_parameters, _results);

    if (!syncRequestImpl)
        syncRequestImpl = CreateInferRequestImpl(_networkInputs, _networkOutputs);
    syncRequestImpl->setPointerToExecutableNetworkInternal(shared_from_this());
    return std::make_shared<MultiDeviceAsyncInferRequest>(std::static_pointer_cast<MultiDeviceInferRequest>(syncRequestImpl),
                                                          _needPerfCounters,
                                                          std::static_pointer_cast<MultiDeviceExecutableNetwork>(shared_from_this()),
                                                          _callbackExecutor);
}

void MultiDeviceExecutableNetwork::SetConfig(const std::map<std::string, InferenceEngine::Parameter> &config) {
    if (_workModeIsAUTO) {
        IE_THROW(NotImplemented);
    }

    auto priorities = config.find(MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES);
    if (priorities == config.end() || config.size() > 1) {
        IE_THROW() << "The only config supported for the Network's SetConfig is MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES";
    } else {
        auto multiPlugin = std::dynamic_pointer_cast<MultiDeviceInferencePlugin>(this->_plugin);
        assert(multiPlugin != nullptr);
        auto metaDevices = multiPlugin->ParseMetaDevices(priorities->second, {});

        if (std::any_of(metaDevices.begin(), metaDevices.end(), [](const DeviceInformation& kvp) {
                return kvp.numRequestsPerDevices != -1;
            })) {
            IE_THROW() << "You can only change device priorities but not number of requests"
                     <<" with the Network's SetConfig(MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES!";
        }

        {
            std::lock_guard<std::mutex> lock{_mutex};
            for (auto && device : metaDevices) {
                if (_networksPerDevice.find(device.deviceName) == _networksPerDevice.end()) {
                    IE_THROW(NotFound) << "You can only change device priorities but not add new devices with"
                        << " the Network's SetConfig(MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES. "
                        << device.deviceName << " device was not in the original device list!";
                }
            }
            _devicePriorities = metaDevices;

            // update value in config
            std::lock_guard<std::mutex> lockConf(_confMutex);
            _config[MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES] = priorities->second;
        }
    }
}

InferenceEngine::Parameter MultiDeviceExecutableNetwork::GetConfig(const std::string &name) const {
    {
        std::lock_guard<std::mutex> lock(_confMutex);
        auto it = _config.find(name);
        if (it != _config.end()) {
            return it->second;
        }
    }

    // find config key among networks config keys
    for (const auto& desc : _networksPerDevice) {
        const auto& execNetwork = desc.second;
        auto param = execNetwork->GetMetric(METRIC_KEY(SUPPORTED_CONFIG_KEYS));
        for (auto &&configKey : param.as<std::vector<std::string>>()) {
            if (configKey == name) {
                return execNetwork->GetConfig(configKey);
            }
        }
        IE_THROW() << "Unsupported ExecutableNetwork config key: " << name;
    }
    IE_THROW(NotFound) << name <<" not found in the ExecutableNetwork config";
}

InferenceEngine::Parameter MultiDeviceExecutableNetwork::GetMetric(const std::string &name) const {
    if (_workModeIsAUTO) {
        if (name == METRIC_KEY(OPTIMAL_NUMBER_OF_INFER_REQUESTS)) {
            unsigned int real = 0;
            if (_loadContext[ACTUALDEVICE].isAlready) {
                real = _loadContext[ACTUALDEVICE].
                    executableNetwork->GetMetric(name).as<unsigned int>();
            } else {
                IE_ASSERT(_loadContext[CPU].isAlready == true);
                real = _loadContext[CPU].
                    executableNetwork->GetMetric(name).as<unsigned int>();
                std::unique_lock<std::mutex> lock(_confMutex);
                auto deviceInfo =  _loadContext[ACTUALDEVICE].deviceInfo;
                lock.unlock();
                if (deviceInfo.deviceName.find("GPU") != std::string::npos) {
                    const auto& mode = deviceInfo.config.find(CONFIG_KEY(PERFORMANCE_HINT));
                    if (mode != deviceInfo.config.end() && mode->second == CONFIG_VALUE(THROUGHPUT)) {
                         std::map<std::string, InferenceEngine::Parameter> options;
                         options["MODEL_PTR"] = _network.getFunction(); // CNNntework
                         try {
                             auto optimalBatchSize = _core->GetMetric(deviceInfo.deviceName,
                                     METRIC_KEY(OPTIMAL_BATCH_SIZE), options).as<unsigned int>();
                             auto rangeOfStreams = _core->GetMetric(deviceInfo.deviceName,
                                     METRIC_KEY(RANGE_FOR_STREAMS), options).as<std::tuple<unsigned int, unsigned int>>();
                             real = (std::max)(real, std::get<1>(rangeOfStreams) * optimalBatchSize);
                         } catch (const InferenceEngine::Exception &iie) {
                             LOG_WARNING("[AUTOPLUGIN]get optimal infer requset num for GPU auto-batch failed :%s", iie.what());
                         }
                    }
                }
            }
            unsigned int res = (std::max)(8u, real);
            IE_SET_METRIC_RETURN(OPTIMAL_NUMBER_OF_INFER_REQUESTS, res);
        }

        if (_loadContext[ACTUALDEVICE].isAlready) {
            return _loadContext[ACTUALDEVICE].executableNetwork->GetMetric(name);
        }
        return _loadContext[CPU].executableNetwork->GetMetric(name);
    }

    if (name == METRIC_KEY(OPTIMAL_NUMBER_OF_INFER_REQUESTS)) {
        unsigned int res = 0u;
        for (auto n : _networksPerDevice) {
            try {
                res += n.second->GetMetric(METRIC_KEY(OPTIMAL_NUMBER_OF_INFER_REQUESTS)).as<unsigned int>();
            } catch (const InferenceEngine::Exception &iie) {
                  IE_THROW()
                        << "Every device used with the Multi-Device should "
                        << "support OPTIMAL_NUMBER_OF_INFER_REQUESTS ExecutableNetwork metric. "
                        << "Failed to query the metric for the " << n.first << " with error:" << iie.what();
           }
        }
        IE_SET_METRIC_RETURN(OPTIMAL_NUMBER_OF_INFER_REQUESTS, res);
    } else if (name == METRIC_KEY(NETWORK_NAME)) {
        auto it = _networksPerDevice.begin();
        IE_ASSERT(it != _networksPerDevice.end());
        IE_SET_METRIC_RETURN(NETWORK_NAME, it->second->GetMetric(
            METRIC_KEY(NETWORK_NAME)).as<std::string>());
    } else if (name == METRIC_KEY(SUPPORTED_METRICS)) {
        IE_SET_METRIC_RETURN(SUPPORTED_METRICS, {
            METRIC_KEY(OPTIMAL_NUMBER_OF_INFER_REQUESTS),
            METRIC_KEY(SUPPORTED_METRICS),
            METRIC_KEY(NETWORK_NAME),
            METRIC_KEY(SUPPORTED_CONFIG_KEYS)
        });
    } else if (name == METRIC_KEY(SUPPORTED_CONFIG_KEYS)) {
        std::vector<std::string> configKeys = { MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES };
        IE_SET_METRIC_RETURN(SUPPORTED_CONFIG_KEYS, configKeys);
    } else {
        IE_THROW() << "Unsupported ExecutableNetwork metric key: " << name;
    }
}
}  // namespace MultiDevicePlugin
