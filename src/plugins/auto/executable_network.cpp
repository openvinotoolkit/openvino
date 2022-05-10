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
    _cpuHelpReleaseTime = std::chrono::steady_clock::now();
    _taskExecutor.reset();
    for (auto&& networkValue : _networksPerDevice) {
        auto& device  = networkValue.first;
        auto& network = networkValue.second;
        GenerateWorkers(device, network);
    }
    if (_networksPerDevice.size() == 1)
        _passthroughExeNet = _networksPerDevice.begin()->second;
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
                    do {
                        _inferPipelineTasks.try_pop(t);
                    } while (t && ScheduleToWorkerInferRequest(std::move(t)));
                    do {
                        _inferPipelineTasksDeviceSpecific[device]->try_pop(t);
                    } while (t && ScheduleToWorkerInferRequest(std::move(t), device));
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
    LOG_INFO("[AUTOPLUGIN]ExecutableNetwork start");
    // initialize cpuHelpReleasetime
    _cpuHelpReleaseTime = std::chrono::steady_clock::now();
    if (_multiPlugin->GetCore() == nullptr) {
        IE_THROW() << "Please, work with " << _multiPlugin->GetName() << " device via InferencEngine::Core object";
    }

    if (modelPath.empty() && network.getFunction() == nullptr) {
        IE_THROW() << "MULTI " << _multiPlugin->GetName() << " device supports just ngraph network representation";
    }

    _core = _multiPlugin->GetCore(); // shared_ptr that holds the Core
    _config[MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES] = strDevices;
    std::string profilingTask = "MultiDeviceExecutableNetwork::MultiDeviceExecutableNetwork:AutoMode";
    bool isCumulative = (context.performanceHint == PluginConfigParams::CUMULATIVE_THROUGHPUT) ? true : false;

    // loadContext[ACTUALDEVICE] is always enabled,
    // when there is CPU and there are more than two devices, loadContext[CPU] is enabled
    _loadContext[ACTUALDEVICE].isEnabled = true;
    _loadContext[ACTUALDEVICE].networkPrecision = GetNetworkPrecision(network);
    _loadContext[ACTUALDEVICE].metaDevices = metaDevices;

    if (isCumulative) {
        std::list<DeviceInformation> validDevices =
            _multiPlugin->GetValidDevice(metaDevices, _loadContext[ACTUALDEVICE].networkPrecision);

        // check if device priority is enabled
        bool enableDevicePriority =
            std::find_if(std::begin(validDevices), std::end(validDevices), [](DeviceInformation& di) {
                return di.devicePriority > 0;
            }) != std::end(validDevices);

        // for the case of -d "AUTO" or "AUTO: -xxx"
        if (!enableDevicePriority) {
            std::list<DeviceInformation>::iterator itCPUDevice;
            int GPUNums = 0, CPUNums = 0;
            for (auto it = validDevices.begin(); it != validDevices.end(); it++) {
                if (it->deviceName.find("GPU") != std::string::npos) {
                    GPUNums++;
                }

                if (it->deviceName.find("CPU") == 0) {
                    CPUNums++;
                    itCPUDevice = it;
                }
            }

            // remove CPU from default candidate list for Cumulative Throughput mode
            if (GPUNums >= 2 && CPUNums > 0) {
                validDevices.erase(itCPUDevice);
                LOG_INFO("[AUTOPLUGIN]:GPUNums:%d, remove CPU from default candidate list for "
                         "CUMULATIVE_THROUGHPUT",
                         GPUNums);
            }
        }

        std::string deviceName = "MULTI:";
        for (auto& device : validDevices) {
            deviceName += device.deviceName;
            deviceName += ((device.deviceName == validDevices.back().deviceName) ? "" : ",");
        }

        _loadContext[ACTUALDEVICE].deviceInfo.deviceName = deviceName;
        _loadContext[ACTUALDEVICE].deviceInfo.config[CONFIG_KEY(PERFORMANCE_HINT)] =
            InferenceEngine::PluginConfigParams::THROUGHPUT;
    } else {
        _loadContext[ACTUALDEVICE].deviceInfo = _multiPlugin->SelectDevice(metaDevices,
                                                                           _loadContext[ACTUALDEVICE].networkPrecision,
                                                                           _context.modelPriority);
    }

    LOG_INFO("[AUTOPLUGIN]:select device:%s", _loadContext[ACTUALDEVICE].deviceInfo.deviceName.c_str());
    bool isActualDevCPU =
        _loadContext[ACTUALDEVICE].deviceInfo.deviceName.find("CPU") != std::string::npos;

    // if Actual device is CPU or hint is cumulative, disabled _loadContext[CPU], only use _loadContext[ACTUALDEVICE]
    if (isActualDevCPU || isCumulative) {
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
             _loadContext[i].task = [this, contextPtr, modelPath, network, isCumulative]() mutable {
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

                          if (!isCumulative) {
                              auto supported_config_keys =
                                  _core->GetMetric(deviceName, METRIC_KEY(SUPPORTED_CONFIG_KEYS))
                                      .as<std::vector<std::string>>();
                              DEBUG_RUN([this, &contextPtr, &deviceName, &supported_config_keys] {
                                  std::lock_guard<std::mutex> lock(_confMutex);
                                  for (const auto& cfg : supported_config_keys) {
                                      try {
                                          LOG_DEBUG(
                                              "[AUTOPLUGIN]:device:%s, GetConfig:%s=%s",
                                              deviceName.c_str(),
                                              cfg.c_str(),
                                              contextPtr->executableNetwork->GetConfig(cfg).as<std::string>().c_str());
                                      } catch (...) {
                                      }
                                  }
                              });
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
        _executor = _multiPlugin->executorManager()->getIdleCPUStreamsExecutor(
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
                std::list<Time> cpuHelpAllStartTimes;
                std::list<Time> cpuHelpAllEndTimes;
                while (_idleWorkerRequests["CPU_HELP"].try_pop(worker)) {
                    destroynum++;
                    INFO_RUN([&cpuHelpAllStartTimes, &cpuHelpAllEndTimes, &worker]() {
                        cpuHelpAllStartTimes.splice(cpuHelpAllStartTimes.end(), worker.second->_startTimes);
                        cpuHelpAllEndTimes.splice(cpuHelpAllEndTimes.end(), worker.second->_endTimes);
                    });
                }
                INFO_RUN([this, &cpuHelpAllStartTimes, &cpuHelpAllEndTimes]() {
                    cpuHelpAllStartTimes.sort(std::less<Time>());
                    cpuHelpAllEndTimes.sort(std::less<Time>());
                    _cpuHelpInferCount = cpuHelpAllStartTimes.size();
                    IE_ASSERT(_cpuHelpInferCount == cpuHelpAllEndTimes.size());
                });
                if (destroynum == _workerRequests["CPU_HELP"].size()) {
                    std::lock_guard<std::mutex> lock(_confMutex);
                    INFO_RUN([this, &cpuHelpAllStartTimes, &cpuHelpAllEndTimes, &destroynum]() {
                        _cpuHelpReleaseTime = std::chrono::steady_clock::now();
                        if (cpuHelpAllStartTimes.size() >= destroynum + 1) {
                            //remove last worksize num requests, so the fps will be more accuracy
                            cpuHelpAllStartTimes.resize(_cpuHelpInferCount - destroynum);
                            cpuHelpAllEndTimes.resize(_cpuHelpInferCount - destroynum);
                            std::chrono::duration<double, std::milli> durtation =
                                cpuHelpAllEndTimes.back() - cpuHelpAllStartTimes.front();
                            _cpuHelpFps = cpuHelpAllStartTimes.size() * 1000 / durtation.count();
                        }
                    });
                    LOG_INFO("[AUTOPLUGIN] release all work requests of CPU_HELP");
                    _workerRequests["CPU_HELP"].clear();
                    _loadContext[CPU].executableNetwork._ptr.reset();
                    _loadContext[CPU].executableNetwork._so.reset();
                    LOG_INFO("[AUTOPLUGIN]:helper released!!");
                    break;
                }
            }
        };
        _executor->run(std::move(recycleTask));
    } else {
        // only one device need to load network, do not need to load it async
        _loadContext[ACTUALDEVICE].task();
        _passthroughExeNet = _loadContext[ACTUALDEVICE].executableNetwork;
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
    bool curDevIsGPU = (device.find("GPU") != std::string::npos);
    {
        std::lock_guard<std::mutex> lock(_confMutex);
        if (curDevIsGPU && _loadContext[CPU].isEnabled) {
            // user does not set the compiling threads
            // limit the threads num for compiling
            int maxNumThreads = 0;
            try {
                maxNumThreads = _core->GetConfig(device, GPU_CONFIG_KEY(MAX_NUM_THREADS)).as<int>();
            } catch (...) {
                LOG_DEBUG("[AUTOPLUGIN]: cannot get MAX_NUM_THREADS from GPU");
            }
            if (maxNumThreads == static_cast<int>(std::thread::hardware_concurrency())) {
                int threadNum = maxNumThreads / 2;
                deviceConfig[GPU_CONFIG_KEY(MAX_NUM_THREADS)] = std::to_string(threadNum).c_str();
                LOG_DEBUG("[AUTO PLUGIN]:gpu streams number for compiling: %s", deviceConfig[GPU_CONFIG_KEY(MAX_NUM_THREADS)].c_str());
            } else {
                // user set the compiling threads num
                // use the user's val anyway
                LOG_DEBUG("[AUTOPLUGIN]:user defined compiling threads: %d", maxNumThreads);
            }
        }
    }
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
        std::lock_guard<std::mutex> lock(_confMutex);
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

bool MultiDeviceExecutableNetwork::ScheduleToWorkerInferRequest(Task inferPipelineTask, DeviceName preferred_device) {
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
            return true;
        }
    }

    // no vacant requests this time, storing the task to the respective queue
    if (!preferred_device.empty())
        _inferPipelineTasksDeviceSpecific[preferred_device]->push(std::move(inferPipelineTask));
    else
        _inferPipelineTasks.push(std::move(inferPipelineTask));
    return false;
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
            _multiPlugin->executorManager()->clear("AutoDeviceAsyncLoad");
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
    INFO_RUN([this] {
        for (auto&& _workerRequest : _workerRequests) {
            std::list<Time> reqAllStartTimes;
            std::list<Time> reqAllEndTimes;
            for (auto& request : _workerRequest.second) {
                reqAllStartTimes.splice(reqAllStartTimes.end(), request._startTimes);
                reqAllEndTimes.splice(reqAllEndTimes.end(), request._endTimes);
            }
            unsigned int count = reqAllStartTimes.size();
            IE_ASSERT(count == reqAllEndTimes.size());
            reqAllStartTimes.sort(std::less<Time>());
            reqAllEndTimes.sort(std::less<Time>());
            if (_workerRequest.first == "CPU_HELP") {
                LOG_INFO("[AUTOPLUGIN]CPU_HELP:infer:%ld", _cpuHelpInferCount + count);
                if (_cpuHelpFps > 0.0) {
                    LOG_INFO("[AUTOPLUGIN]CPU_HELP:fps:%lf", _cpuHelpFps);
                } else if (count >= 1) {
                    std::chrono::duration<double, std::milli> durtation =
                        reqAllEndTimes.back() - reqAllStartTimes.front();
                    LOG_INFO("[AUTOPLUGIN]CPU_HELP:fps:%lf", count * 1000 / durtation.count());
                }
            } else {
                LOG_INFO("[AUTOPLUGIN]%s:infer:%ld", _workerRequest.first.c_str(), count);
                auto n = reqAllStartTimes.size();
                Time time;
                while (!reqAllStartTimes.empty()) {
                    time = reqAllStartTimes.front();
                    if (time < _cpuHelpReleaseTime) {
                        reqAllStartTimes.pop_front();
                        n--;
                    } else {
                        break;
                    }
                }
                if (n >= 1) {
                    std::chrono::duration<double, std::milli> durtation =
                        reqAllEndTimes.back() - time;
                    LOG_INFO("[AUTOPLUGIN]%s:fps:%lf", _workerRequest.first.c_str(),
                        n * 1000 / durtation.count());
                }
            }
        }
    });
    {
        std::lock_guard<std::mutex> lock(_confMutex);
        _workerRequests.clear();
    }
    LOG_INFO("[AUTOPLUGIN]ExecutableNetwork end");
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
        } catch (const InferenceEngine::NotImplemented&) {}
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
    InferenceEngine::SoIInferRequestInternal request_to_share_blobs_with;
    InferenceEngine::RemoteContext::Ptr ctx = nullptr;

    if (_workModeIsAUTO) {
        if (!_loadContext[CPU].isEnabled && _loadContext[ACTUALDEVICE].isAlready) {
            try {
                ctx = GetCore()->GetDefaultContext(_loadContext[ACTUALDEVICE].deviceInfo.deviceName);
            } catch (InferenceEngine::Exception& ex) {
                // plugin does not support context, say CPU
                LOG_DEBUG("[AUTOPLUGIN]context not supported for %s, fallback to default memory",
                                _loadContext[ACTUALDEVICE].deviceInfo.deviceName.c_str());
                // for dynamic shape support
                auto& dev_requests = _workerRequests[_loadContext[ACTUALDEVICE].deviceInfo.deviceName];
                if (num < dev_requests.size()) {
                    request_to_share_blobs_with = dev_requests.at(num)._inferRequest;
                }
            }
        }
        return std::make_shared<MultiDeviceInferRequest>(inputs, outputs, request_to_share_blobs_with, ctx);
    }

    return std::make_shared<MultiDeviceInferRequest>(inputs, outputs, request_to_share_blobs_with);
}

InferenceEngine::IInferRequestInternal::Ptr MultiDeviceExecutableNetwork::CreateInferRequestImpl(InferenceEngine::InputsDataMap networkInputs,
                                                                                                InferenceEngine::OutputsDataMap networkOutputs) {
    auto num = _numRequestsCreated++;
    InferenceEngine::SoIInferRequestInternal request_to_share_blobs_with;
    InferenceEngine::RemoteContext::Ptr ctx = nullptr;

    if (_workModeIsAUTO) {
        if (!_loadContext[CPU].isEnabled && _loadContext[ACTUALDEVICE].isAlready) {
            try {
                ctx = GetCore()->GetDefaultContext(_loadContext[ACTUALDEVICE].deviceInfo.deviceName);
            } catch (InferenceEngine::Exception& ex) {
                // plugin does not support context
                LOG_DEBUG("[AUTOPLUGIN]context not supported for %s, fallback to default memory",
                                _loadContext[ACTUALDEVICE].deviceInfo.deviceName.c_str());
                auto& dev_requests = _workerRequests[_loadContext[ACTUALDEVICE].deviceInfo.deviceName];
                if (num < dev_requests.size()) {
                    request_to_share_blobs_with = dev_requests.at(num)._inferRequest;
                }
            }
        }
        return std::make_shared<MultiDeviceInferRequest>(networkInputs, networkOutputs, request_to_share_blobs_with, ctx);
    }

    return std::make_shared<MultiDeviceInferRequest>(networkInputs, networkOutputs, request_to_share_blobs_with);
}

IInferRequestInternal::Ptr MultiDeviceExecutableNetwork::CreateInferRequest() {
    if (_passthroughExeNet) {
        auto res = _passthroughExeNet->CreateInferRequest();
        res->setPointerToExecutableNetworkInternal(shared_from_this());
        return res;
    }
    IInferRequestInternal::Ptr syncRequestImpl;
    if (this->_plugin) {
        const auto& core = _plugin->GetCore();
        if (core && core->isNewAPI())
            syncRequestImpl = CreateInferRequestImpl(_parameters, _results);
    }

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

    auto priorities = config.find(ov::device::priorities.name());
    if (priorities == config.end() || config.size() > 1) {
        IE_THROW() << "The only config supported for the Network's SetConfig is MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES";
    } else {
        auto multiPlugin = std::dynamic_pointer_cast<MultiDeviceInferencePlugin>(this->_plugin);
        assert(multiPlugin != nullptr);
        auto metaDevices = multiPlugin->ParseMetaDevices(priorities->second.as<std::string>(), {});

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
        if (name == ov::supported_properties) {
            return decltype(ov::supported_properties)::value_type {
                // Metrics
                ov::PropertyName{ov::supported_properties.name(), ov::PropertyMutability::RO},
                ov::PropertyName{ov::hint::performance_mode.name(), ov::PropertyMutability::RO},
                ov::PropertyName{ov::model_name.name(), ov::PropertyMutability::RO},
                ov::PropertyName{ov::optimal_number_of_infer_requests.name(), ov::PropertyMutability::RO},
                ov::PropertyName{ov::hint::model_priority.name(), ov::PropertyMutability::RO},
                ov::PropertyName{ov::device::priorities.name(), ov::PropertyMutability::RO}
            };
        } else if (name == ov::device::priorities) {
            auto value = _config.find(MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES);
            return decltype(ov::device::priorities)::value_type {value->second.as<std::string>()};
        } else if (name == ov::hint::model_priority) {
            auto value = _context.modelPriority;
            if (_core->isNewAPI()) {
                return value ? ((value > 1) ? ov::hint::Priority::LOW : ov::hint::Priority::MEDIUM) : ov::hint::Priority::HIGH;
            } else {
                return value ? ((value > 1) ? CONFIG_VALUE(MODEL_PRIORITY_LOW) : CONFIG_VALUE(MODEL_PRIORITY_MED)) : CONFIG_VALUE(MODEL_PRIORITY_HIGH);
            }
        } else if (name == ov::optimal_number_of_infer_requests) {
            const unsigned int defaultNumForTPUT = 4u;
            const unsigned int defaultNumForLatency = 1u;
            unsigned int real = 0;
            if (_loadContext[ACTUALDEVICE].isAlready) {
                real = _loadContext[ACTUALDEVICE].
                    executableNetwork->GetMetric(name).as<unsigned int>();
            } else {
                IE_ASSERT(_loadContext[CPU].isAlready == true);
                std::unique_lock<std::mutex> lock(_confMutex);
                auto deviceInfo =  _loadContext[ACTUALDEVICE].deviceInfo;
                lock.unlock();
                unsigned int optimalBatchSize = 0;
                unsigned int requests = 0;
                bool bThroughputEnabledInPlugin = false;
                try {
                    // for benchmark through AUTO:CPU,GPU
                    // SetConfig directly set to CPU/GPU in this case
                    bThroughputEnabledInPlugin =
                        _core->GetConfig(deviceInfo.deviceName, CONFIG_KEY(PERFORMANCE_HINT)).as<std::string>() == CONFIG_VALUE(THROUGHPUT);
                } catch (...) {
                    LOG_DEBUG("[AUTOPLUGIN]GetMetric:%s for %s", "PERF_HINT config not supported", deviceInfo.deviceName.c_str());
                }
                const auto& mode = deviceInfo.config.find(CONFIG_KEY(PERFORMANCE_HINT));
                if (bThroughputEnabledInPlugin ||
                    (mode != deviceInfo.config.end() && mode->second == CONFIG_VALUE(THROUGHPUT))) {
                    unsigned int upperBoundStreamsNum = 0;
                    std::map<std::string, InferenceEngine::Parameter> options;
                    options["MODEL_PTR"] = std::const_pointer_cast<ngraph::Function>(_network.getFunction());
                    try {
                        auto rangeOfStreams = _core->GetMetric(deviceInfo.deviceName,
                                                        METRIC_KEY(RANGE_FOR_STREAMS), options).as<std::tuple<unsigned int, unsigned int>>();
                        upperBoundStreamsNum = std::get<1>(rangeOfStreams);
                    } catch (const InferenceEngine::Exception &iie) {
                        LOG_DEBUG("[AUTOPLUGIN] GetMetric RANGE_FOR_STREAMS failed");
                    }
                    if (!_context.batchingDisabled) {
                        try {
                            optimalBatchSize = _core->GetMetric(deviceInfo.deviceName,
                                            METRIC_KEY(OPTIMAL_BATCH_SIZE), options).as<unsigned int>();
                            LOG_DEBUG("[AUTOPLUGIN]BATCHING:%s:%ld", "optimal batch size", optimalBatchSize);
                        } catch (...) {
                            LOG_DEBUG("[AUTOPLUGIN]BATCHING:%s", "metric OPTIMAL_BATCH_SIZE not supported");
                        }
                    }
                    if (optimalBatchSize > 1) {
                        // batching is supported with the device
                        // go with auto-batching
                        try {
                            // check if app have set preferred value
                            auto res =
                                _core->GetConfig(deviceInfo.deviceName, CONFIG_KEY(PERFORMANCE_HINT_NUM_REQUESTS)).as<std::string>();
                            requests = PerfHintsConfig::CheckPerformanceHintRequestValue(res);
                            const auto& reqs = deviceInfo.config.find(CONFIG_KEY(PERFORMANCE_HINT_NUM_REQUESTS));
                            if (reqs != deviceInfo.config.end())
                                requests = static_cast<unsigned int>(PerfHintsConfig::CheckPerformanceHintRequestValue(reqs->second));
                            LOG_DEBUG("[AUTOPLUGIN]BATCHING:%s:%ld", "user requested size", requests);
                            if (!requests) { // no limitations from user
                                requests = optimalBatchSize * upperBoundStreamsNum * 2;
                                LOG_DEBUG("[AUTOPLUGIN]BATCHING:%s:%ld", "deduced size:", requests);
                            }
                        } catch (const InferenceEngine::Exception &iie) {
                            LOG_WARNING("[AUTOPLUGIN]deduce optimal infer requset num for auto-batch failed :%s", iie.what());
                        }
                        real = (std::max)(requests, optimalBatchSize);
                    } else if (deviceInfo.deviceName.find("VPUX") != std::string::npos) {
                        real = 8u;
                    } else {
                        real = upperBoundStreamsNum ? 2 * upperBoundStreamsNum : defaultNumForTPUT;
                    }
                } else {
                    real = defaultNumForLatency;
                }
            }
            return decltype(ov::optimal_number_of_infer_requests)::value_type {real};
        }

        if (_loadContext[ACTUALDEVICE].isAlready) {
            return _loadContext[ACTUALDEVICE].executableNetwork->GetMetric(name);
        }
        return _loadContext[CPU].executableNetwork->GetMetric(name);
    }

    if (name == ov::supported_properties) {
        return decltype(ov::supported_properties)::value_type {
            // Metrics
            ov::PropertyName{ov::supported_properties.name(), ov::PropertyMutability::RO},
            ov::PropertyName{ov::model_name.name(), ov::PropertyMutability::RO},
            ov::PropertyName{ov::optimal_number_of_infer_requests.name(), ov::PropertyMutability::RO},

            // Configs
            // device priority can be changed on-the-fly in MULTI
            ov::PropertyName{ov::device::priorities.name(), ov::PropertyMutability::RW}
        };
    } else if (name == ov::optimal_number_of_infer_requests) {
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
        return decltype(ov::optimal_number_of_infer_requests)::value_type {res};
    } else if (name == ov::model_name) {
        auto it = _networksPerDevice.begin();
        IE_ASSERT(it != _networksPerDevice.end());
        return decltype(ov::model_name)::value_type {it->second->GetMetric(METRIC_KEY(NETWORK_NAME)).as<std::string>()};
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
