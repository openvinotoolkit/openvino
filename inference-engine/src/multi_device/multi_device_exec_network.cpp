// Copyright (C) 2018-2021 Intel Corporation
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
#include "multi_device_exec_network.hpp"
#include "multi_device_async_infer_request.hpp"
#include "multi_device_plugin.hpp"

#include "ngraph/opsets/opset1.hpp"
#include "transformations/utils/utils.hpp"

#include "multi_itt.hpp"
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
            _notBusyWorkerRequests->try_push(_workerInferRequestPtr);
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
    auto itNumRequests = std::find_if(_devicePriorities.cbegin(), _devicePriorities.cend(),
                                      [&device](const DeviceInformation& d){ return d.deviceName == device;});
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
    for (auto&& workerRequest : workerRequests) {
        workerRequest._inferRequest = { executableNetwork._so, executableNetwork->CreateInferRequest() };
        auto* workerRequestPtr = &workerRequest;
        IE_ASSERT(idleWorkerRequests.try_push(workerRequestPtr) == true);
        workerRequest._inferRequest->SetCallback(
            [workerRequestPtr, this, device, idleWorkerRequestsPtr] (std::exception_ptr exceptionPtr) mutable {
                IdleGuard idleGuard{workerRequestPtr, *idleWorkerRequestsPtr};
                workerRequestPtr->_exceptionPtr = exceptionPtr;
                {
                    auto capturedTask = std::move(workerRequestPtr->_task);
                    capturedTask();
                }
                // try to return the request to the idle list (fails if the overall object destruction has began)
                if (idleGuard.Release()->try_push(workerRequestPtr)) {
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

void MultiDeviceExecutableNetwork::IncreaseWorkers(AutoLoadContext& loadcontext, InferenceEngine::SoIInferRequestInternal& request_to_share) {
    auto devicename = loadcontext.deviceInfo.deviceName;
     auto& executableNetwork =  loadcontext.executableNetwork;
    auto& idleWorkerRequests = _idleWorkerRequests[devicename];
    auto* idleWorkerRequestsPtr = &(idleWorkerRequests);
    {
        std::lock_guard<std::mutex> lock(_idleWorkerMutex);
        idleWorkerRequests.set_capacity(idleWorkerRequests.capacity() + 1);
    }
    auto* workerRequestPtr =  new WorkerInferRequest;
    workerRequestPtr->_manualyDestory = true;
    auto& workerRequest = *workerRequestPtr;
    workerRequest._inferRequest = { executableNetwork._so, executableNetwork->CreateInferRequest() };
    IE_ASSERT(idleWorkerRequests.try_push(workerRequestPtr) == true);
    workerRequest._inferRequest->SetCallback(
            [workerRequestPtr, this, devicename, idleWorkerRequestsPtr] (std::exception_ptr exceptionPtr) mutable {
                IdleGuard idleGuard{workerRequestPtr, *idleWorkerRequestsPtr};
                workerRequestPtr->_exceptionPtr = exceptionPtr;
                {
                    auto capturedTask = std::move(workerRequestPtr->_task);
                    capturedTask();
                }
                // try to return the request to the idle list (fails if the overall object destruction has began)
                if (idleGuard.Release()->try_push(workerRequestPtr)) {
                    // let's try to pop a task, as we know there is at least one idle request, schedule if succeeded
                    // if no device-agnostic tasks, let's try pop the device specific task, schedule if succeeded
                    Task t;
                    if (_inferPipelineTasks.try_pop(t))
                        ScheduleToWorkerInferRequest(std::move(t));
                    else if (_inferPipelineTasksDeviceSpecific[devicename]->try_pop(t))
                        ScheduleToWorkerInferRequest(std::move(t), devicename);
                }
            });
    request_to_share = workerRequest._inferRequest;
}

MultiDeviceExecutableNetwork::MultiDeviceExecutableNetwork(const std::string&                         modelPath,
                                                           const InferenceEngine::CNNNetwork&         network,
                                                           const std::vector<DeviceInformation>&      metaDevices,
                                                           const std::string&                         strDevices,
                                                           MultiDeviceInferencePlugin*                plugin,
                                                           const bool                                needPerfCounters)
                                                           : _devicePriorities{metaDevices}
                                                           , _devicePrioritiesInitial{metaDevices}
                                                           , _needPerfCounters(needPerfCounters)
                                                           , _multiPlugin(plugin)
                                                           , _workModeIsAUTO(true) {
    if (_multiPlugin->GetCore() == nullptr) {
        IE_THROW() << "Please, work with MULTI device via InferencEngine::Core object";
    }

    if (modelPath.empty() && network.getFunction() == nullptr) {
        IE_THROW() << "MULTI device supports just ngraph network representation";
    }

    _core = _multiPlugin->GetCore(); // shared_ptr that holds the Core
    _config[MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES] = strDevices;

    std::string profilingTask = "MultiDeviceExecutableNetwork::MultiDeviceExecutableNetwork:AutoMode";

    // loadContext[ACTUALDEVICE] is always enabled,
    // when there is CPU and there are more than two devices, loadContext[CPU] is enabled
    _loadContext[ACTUALDEVICE].isEnabled = true;
    _loadContext[ACTUALDEVICE].networkPrecision = GetNetworkPrecision(network);
    _loadContext[ACTUALDEVICE].metaDevices = metaDevices;
    _loadContext[ACTUALDEVICE].deviceInfo = _multiPlugin->SelectDevice(metaDevices, _loadContext[ACTUALDEVICE].networkPrecision);
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
                          GenerateWorkers(contextPtr->deviceInfo.deviceName, contextPtr->executableNetwork);
                          //need lock
                          {
                             std::lock_guard<std::mutex> lock(_confMutex);
                             _config.insert(contextPtr->deviceInfo.config.begin(),
                                            contextPtr->deviceInfo.config.end());
                          }
                          contextPtr->isAlready = true;
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
        _executor->run(_loadContext[CPU].task);
        _executor->run(_loadContext[ACTUALDEVICE].task);
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
        context.deviceInfo = _multiPlugin->SelectDevice(deviceList, context.networkPrecision);
    }
    catch (const std::exception& e) {
        return;
    }

    // if selec device is CPU, do not need to load CPU again, context[CPU] must have loaded CPU
    curDevIsCPU = (context.deviceInfo.deviceName.find("CPU") != std::string::npos);
    if (curDevIsCPU) {
        return;
    }

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

    // ToDo need to print failed error mesage
    IE_THROW() << "[AUTO] load all devices failed";
}

void MultiDeviceExecutableNetwork::WaitActualNetworkReady() const {
    OV_ITT_SCOPED_TASK(itt::domains::MULTIPlugin, "MultiDeviceExecutableNetwork::WaitActualNetworkReady");
    // Maybe different API will call this function, so add call once here
    // for every MultiDeviceExecutableNetwork instance
    std::call_once(_oc, [this] () {
               if (_loadContext[ACTUALDEVICE].future.valid()) {
                   _loadContext[ACTUALDEVICE].future.get();
               }
               // if _loadContext[ACTUALDEVICE] load failed,  fall back to _loadContext[CPU]
               if (!_loadContext[ACTUALDEVICE].isAlready) {
                   _loadContext[ACTUALDEVICE].executableNetwork = _loadContext[CPU].executableNetwork;
                   _loadContext[ACTUALDEVICE].deviceInfo = _loadContext[CPU].deviceInfo;
                   _loadContext[ACTUALDEVICE].isAlready = true;
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
                IE_THROW(NotFound) << "The preferred_device should be the selected device";
            }
            devices.push_back(_loadContext[ACTUALDEVICE].deviceInfo);
        } else {
            if (_loadContext[ACTUALDEVICE].isAlready) {
                devices.push_back(_loadContext[ACTUALDEVICE].deviceInfo);
            } else {
                devices.push_back(_loadContext[CPU].deviceInfo);
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
  if (idleWorkerRequests.try_pop(workerRequestPtr)) {
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
    // this is necessary to guarantee member destroyed after getting future
    if (_workModeIsAUTO && _loadContext[CPU].isEnabled) {
        _loadContext[CPU].future.get();
        WaitActualNetworkReady();
        // it's necessary to wait the loading network threads to stop here.
        InferenceEngine::ExecutorManager::getInstance()->clear("AutoDeviceAsyncLoad");
        _executor.reset();
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
    if (_workModeIsAUTO) {
        for (auto&& idleWorker : _idleWorkerRequests) {
            WorkerInferRequest *workerRequestPtr = nullptr;
            if (idleWorker.second.try_pop(workerRequestPtr)) {
                if (workerRequestPtr->_manualyDestory) {
                    delete workerRequestPtr;
                }
            }
        }
    }
    _workerRequests.clear();
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
    IE_THROW(NotImplemented) << "None of the devices in the MULTI has an associated remote context."
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
            } else {
                IncreaseWorkers(_loadContext[ACTUALDEVICE], request_to_share_blobs_with);
            }
        } else {
            auto& dev_requests = _workerRequests[_loadContext[CPU].deviceInfo.deviceName];
            if (num < dev_requests.size()) {
                request_to_share_blobs_with = dev_requests.at(num)._inferRequest;
            } else {
                IncreaseWorkers(_loadContext[CPU], request_to_share_blobs_with);
            }
        }
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
            } else {
                IncreaseWorkers(_loadContext[ACTUALDEVICE], request_to_share_blobs_with);
            }
        } else {
            auto& dev_requests = _workerRequests[_loadContext[CPU].deviceInfo.deviceName];
            if (num < dev_requests.size()) {
                request_to_share_blobs_with = dev_requests.at(num)._inferRequest;
            } else {
                IncreaseWorkers(_loadContext[CPU], request_to_share_blobs_with);
            }
        }
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
            _confMutex.lock();
            _config[MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES] = priorities->second;
            _confMutex.unlock();
        }
    }
}

InferenceEngine::Parameter MultiDeviceExecutableNetwork::GetConfig(const std::string &name) const {
    _confMutex.lock();
    auto it = _config.find(name);
    if (it != _config.end()) {
        _confMutex.unlock();
        return it->second;
    } else {
        _confMutex.unlock();
        // find config key among networks config keys
        for (const auto& desc : _networksPerDevice) {
            const auto& execNetwork = desc.second;
            auto param = execNetwork->GetMetric(METRIC_KEY(SUPPORTED_CONFIG_KEYS));
            for (auto &&configKey : param.as<std::vector<std::string>>()) {
                if (configKey == name) {
                    return execNetwork->GetConfig(configKey);
                }
            }
        }
        IE_THROW(NotFound) << name <<" not found in the ExecutableNetwork config";
    }
}

InferenceEngine::Parameter MultiDeviceExecutableNetwork::GetMetric(const std::string &name) const {
    if (_workModeIsAUTO) {
        // fixme: should we wait actual device? meanwhile it will block inference, how to fix?
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
        IE_THROW() << "Unsupported Network metric: " << name;
    }
}
}  // namespace MultiDevicePlugin
