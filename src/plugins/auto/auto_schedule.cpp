// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///////////////////////////////////////////////////////////////////////////////////////////////////

#include "auto_schedule.hpp"
#include "async_infer_request.hpp"
#include "auto_executable_network.hpp"
#include "plugin.hpp"

// ------------------------------AutoSchedule----------------------------
namespace MultiDevicePlugin {

namespace {
std::string GetNetworkPrecision(const IE::CNNNetwork& network) {
    auto nGraphFunc = network.getFunction();
    bool isINTModel = ngraph::op::util::has_op_with_type<ngraph::op::FakeQuantize>
        (nGraphFunc);
    if (isINTModel) {
        return METRIC_VALUE(INT8);
    }
    for (auto& node : nGraphFunc->get_ordered_ops()) {
        if (std::dynamic_pointer_cast<ngraph::opset1::Convolution>(node) ||
            std::dynamic_pointer_cast<ngraph::opset1::GroupConvolution>(node) ||
            std::dynamic_pointer_cast<ngraph::opset1::GroupConvolutionBackpropData>(node) ||
            std::dynamic_pointer_cast<ngraph::opset1::ConvolutionBackpropData>(node)) {
            auto layerType = node->input(1).get_element_type().get_type_name();
            if (layerType == "f32") {
                return METRIC_VALUE(FP32);
            }
            if (layerType == "f16") {
                return METRIC_VALUE(FP16);
            }
        }
    }
    return METRIC_VALUE(FP32);
}
}  // namespace

void AutoSchedule::GenerateWorkers(const std::string& device,
    const SoExecNetwork& executableNetwork) {
    std::string realDeviceName;
    if (device == "CPU_HELP") {
        realDeviceName = "CPU";
    } else {
        realDeviceName = device;
    }
    auto itNumRequests = std::find_if(_autoSContext->_devicePriorities.cbegin(), _autoSContext->_devicePriorities.cend(),
                                      [&realDeviceName](const DeviceInformation & d) {
                                          return d.deviceName == realDeviceName;
                                      });
    unsigned int optimalNum = 0;
    try {
        optimalNum = executableNetwork->GetMetric(METRIC_KEY(OPTIMAL_NUMBER_OF_INFER_REQUESTS)).as<unsigned int>();
    } catch (const IE::Exception& iie) {
        IE_THROW()
                << "Every device used with the Multi-Device should "
                    << "support OPTIMAL_NUMBER_OF_INFER_REQUESTS ExecutableNetwork metric. "
                    << "Failed to query the metric for the " << device << " with error:" <<
                    iie.what();
    }
    const auto numRequests = (_autoSContext->_devicePriorities.end() == itNumRequests ||
                              itNumRequests->numRequestsPerDevices == -1) ? optimalNum : itNumRequests->numRequestsPerDevices;
    auto& workerRequests = _workerRequests[device];
    auto& idleWorkerRequests = _idleWorkerRequests[device];
    workerRequests.resize(numRequests);
    _inferPipelineTasksDeviceSpecific[device] = std::unique_ptr<IE::ThreadSafeQueue<IE::Task>>(new IE::ThreadSafeQueue<IE::Task>);
    auto* idleWorkerRequestsPtr = &(idleWorkerRequests);
    idleWorkerRequests.set_capacity(numRequests);
    int num = 0;
    for (auto&& workerRequest : workerRequests) {
        workerRequest._inferRequest = {executableNetwork->CreateInferRequest(), executableNetwork._so};
        auto* workerRequestPtr = &workerRequest;
        workerRequestPtr->_index = num++;
        IE_ASSERT(idleWorkerRequests.try_push(std::make_pair(workerRequestPtr->_index, workerRequestPtr)) == true);
        workerRequest._inferRequest->SetCallback(
            [workerRequestPtr, this, device, idleWorkerRequestsPtr](std::exception_ptr exceptionPtr) mutable {
                IdleGuard<NotBusyPriorityWorkerRequests> idleGuard{workerRequestPtr, *idleWorkerRequestsPtr};
                workerRequestPtr->_exceptionPtr = exceptionPtr;
                {
                    auto capturedTask = std::move(workerRequestPtr->_task);
                    capturedTask();
                }
                // try to return the request to the idle list (fails if the overall object destruction has began)
                if (idleGuard.Release()->try_push(std::make_pair(workerRequestPtr->_index, workerRequestPtr))) {
                    // let's try to pop a task, as we know there is at least one idle request, schedule if succeeded
                    // if no device-agnostic tasks, let's try pop the device specific task, schedule if succeeded
                    IE::Task t;
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

void AutoSchedule::init(const ScheduleContext::Ptr& sContext) {
    _LogTag = sContext->_LogTag;
    LOG_INFO_TAG("ExecutableNetwork start");
    // initialize cpuHelpReleasetime
    _cpuHelpReleaseTime = std::chrono::steady_clock::now();
    _multiSContext = std::dynamic_pointer_cast<MultiScheduleContext>(sContext);
    _autoSContext = std::dynamic_pointer_cast<AutoScheduleContext>(sContext);
    if (_autoSContext->_core == nullptr) {
        IE_THROW() << "Please, work with Auto device via InferencEngine::Core object";
    }
    if (_autoSContext->_modelPath.empty() && _autoSContext->_network.getFunction() == nullptr) {
        IE_THROW() << "AUTO device supports just ngraph network representation";
    }
    _autoSContext->_config[IE::MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES] = _autoSContext->_strDevices;
    std::string profilingTask = "AutoSchedule::AutoSchedule:AutoMode";
    bool isCumulative = (_autoSContext->_performanceHint == IE::PluginConfigParams::CUMULATIVE_THROUGHPUT) ? true : false;
    // loadContext[ACTUALDEVICE] is always enabled,
    // when there is CPU and there are more than two devices, loadContext[CPU] is enabled
    _loadContext[ACTUALDEVICE].isEnabled = true;
    if (_autoSContext->_modelPath.empty())
        _loadContext[ACTUALDEVICE].networkPrecision = GetNetworkPrecision(_autoSContext->_network);
    _loadContext[ACTUALDEVICE].metaDevices = _autoSContext->_devicePriorities;
    if (isCumulative) {
        std::list<DeviceInformation> validDevices =
            _autoSContext->_plugin->GetValidDevice(_autoSContext->_devicePriorities, _loadContext[ACTUALDEVICE].networkPrecision);

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
            if (GPUNums >= 3 && CPUNums > 0 && !_autoSContext->_bindBuffer) {
                validDevices.erase(itCPUDevice);
                LOG_INFO_TAG("GPUNums:%d, remove CPU from default candidate list for "
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
            InferenceEngine::PluginConfigParams::CUMULATIVE_THROUGHPUT;
        _loadContext[ACTUALDEVICE].deviceInfo.config[CONFIG_KEY(PERF_COUNT)] =
            _autoSContext->_needPerfCounters ? InferenceEngine::PluginConfigParams::YES
                                             : InferenceEngine::PluginConfigParams::NO;
        if (_autoSContext->_bindBuffer)
            _loadContext[ACTUALDEVICE].deviceInfo.config[ov::intel_auto::device_bind_buffer.name()] = InferenceEngine::PluginConfigParams::YES;
    } else {
        _loadContext[ACTUALDEVICE].deviceInfo = _autoSContext->_plugin->SelectDevice(_autoSContext->_devicePriorities,
                                                                           _loadContext[ACTUALDEVICE].networkPrecision,
                                                                           _autoSContext->_modelPriority);
    }
    LOG_INFO_TAG("select device:%s", _loadContext[ACTUALDEVICE].deviceInfo.deviceName.c_str());
    bool isActualDevCPU =
        _loadContext[ACTUALDEVICE].deviceInfo.deviceName.find("CPU") !=std::string::npos;
    // if Actual device is CPU, disabled _loadContext[CPU], only use _loadContext[ACTUALDEVICE]
    if (isActualDevCPU || isCumulative) {
        _loadContext[CPU].isEnabled = false;
    } else {
        const auto CPUIter = std::find_if(_autoSContext->_devicePriorities.begin(), _autoSContext->_devicePriorities.end(),
                                          [=](const DeviceInformation& d) -> bool { return d.deviceName.find("CPU") != std::string::npos; });
        // if have CPU Device,  enable _loadContext[CPU]
        if (CPUIter != _autoSContext->_devicePriorities.end()) {
            _loadContext[CPU].isEnabled = true;
            _loadContext[CPU].deviceInfo = *CPUIter;
            _loadContext[CPU].deviceInfo.config[CONFIG_KEY(PERFORMANCE_HINT)] =
                IE::PluginConfigParams::LATENCY;
            _loadContext[CPU].workName = "CPU_HELP";
            LOG_INFO_TAG("will load CPU for accelerator");
        } else {
            _loadContext[CPU].isEnabled = false;
        }
    }
    // initialize the rest members of load context
    for (int i = 0; i < CONTEXTNUM; i++) {
        if (_loadContext[i].isEnabled) {
            _loadContext[i].future =  _loadContext[i].promise.get_future();
            auto* contextPtr = &_loadContext[i];
            auto modelPath = _autoSContext->_modelPath;
            auto network = _autoSContext->_network;
            _loadContext[i].task = [this, contextPtr, modelPath, network, isCumulative]() mutable {
                TryToLoadNetWork(*contextPtr, modelPath, network);
                if (contextPtr->isLoadSuccess) {
                    if (contextPtr->workName.empty()) {
                        contextPtr->workName = contextPtr->deviceInfo.deviceName;
                    }
                    if (!isCumulative)
                        GenerateWorkers(contextPtr->workName, contextPtr->executableNetwork);
                    //need lock
                    {
                        std::lock_guard<std::mutex> lock(_autoSContext->_confMutex);
                        _autoSContext->_config.insert(contextPtr->deviceInfo.config.begin(), contextPtr->deviceInfo.config.end());
                    }
                    contextPtr->isAlready = true;
                    auto& deviceName = contextPtr->deviceInfo.deviceName;
                    LOG_INFO_TAG("device:%s loading Network finished", deviceName.c_str());
                    if (!isCumulative) {
                        auto supported_config_keys =
                            _autoSContext->_core->GetMetric(deviceName, METRIC_KEY(SUPPORTED_CONFIG_KEYS))
                                      .as<std::vector<std::string>>();
                        DEBUG_RUN([this, &contextPtr, &deviceName, &supported_config_keys] {
                            std::lock_guard<std::mutex> lock(_autoSContext->_confMutex);
                            for (const auto& cfg : supported_config_keys) {
                                try {
                                    LOG_DEBUG_TAG(
                                        "device:%s, GetConfig:%s=%s",
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
                std::call_once(_firstLoadOC, [this]() {
                    _firstLoadPromise.set_value();
                });
            };
        }
    }
    OV_ITT_SCOPED_TASK(itt::domains::MULTIPlugin,
        openvino::itt::handle(profilingTask));
    if (_loadContext[CPU].isEnabled) {
        _firstLoadFuture = _firstLoadPromise.get_future();
        // will not wait for loading accelerator network,
        // so the executor can't be destroyed before finished the task,
        // so use executor as a member of AutoSchedule.
        _executor = _autoSContext->_plugin->executorManager()->getIdleCPUStreamsExecutor(
                    IE::IStreamsExecutor::Config{"AutoDeviceAsyncLoad",
                    static_cast<int>(std::thread::hardware_concurrency()) /* max possible #streams*/,
                    0 /*default threads per stream, workaround for ticket 62376*/,
                    IE::IStreamsExecutor::ThreadBindingType::NONE});
        for (auto&& device : _autoSContext->_devicePriorities) {
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
                    iter._inferRequest._ptr->Wait(IE::InferRequest::WaitMode::RESULT_READY);
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
                    std::lock_guard<std::mutex> lock(_autoSContext->_confMutex);
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
                    LOG_INFO_TAG("release all work requests of CPU_HELP");
                    _workerRequests["CPU_HELP"].clear();
                    _loadContext[CPU].executableNetwork._ptr.reset();
                    _loadContext[CPU].executableNetwork._so.reset();
                    LOG_INFO_TAG("helper released!!");
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

void AutoSchedule::TryToLoadNetWork(AutoLoadContext& context, const std::string& modelPath, const IE::CNNNetwork& network) {
    auto& device = context.deviceInfo.deviceName;
    auto& deviceConfig = context.deviceInfo.config;
    auto& deviceList = context.metaDevices;
    bool curDevIsCPU = (device.find("CPU") != std::string::npos);
    bool curDevIsGPU = (device.find("GPU") != std::string::npos);
    {
        std::lock_guard<std::mutex> lock(_autoSContext->_confMutex);
        if (curDevIsGPU && _loadContext[CPU].isEnabled) {
            // user does not set the compiling threads
            // limit the threads num for compiling
            int maxNumThreads = 0;
            try {
                maxNumThreads = _autoSContext->_core->GetConfig(device, GPU_CONFIG_KEY(MAX_NUM_THREADS)).as<int>();
            } catch (...) {
                LOG_DEBUG_TAG("cannot get MAX_NUM_THREADS from GPU");
            }
            if (maxNumThreads == static_cast<int>(std::thread::hardware_concurrency())) {
                int threadNum = maxNumThreads / 2;
                deviceConfig[GPU_CONFIG_KEY(MAX_NUM_THREADS)] = std::to_string(threadNum).c_str();
                LOG_DEBUG_TAG("gpu streams number for compiling: %s",
                          deviceConfig[GPU_CONFIG_KEY(MAX_NUM_THREADS)].c_str());
            } else {
                // user set the compiling threads num
                // use the user's val anyway
                LOG_DEBUG_TAG("user defined compiling threads: %d", maxNumThreads);
            }
        }
    }
    try {
        if (!modelPath.empty()) {
            context.executableNetwork = _autoSContext->_core->LoadNetwork(modelPath, device, deviceConfig);
        } else {
            context.executableNetwork = _autoSContext->_core->LoadNetwork(network, device, deviceConfig);
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
    _autoSContext->_plugin->UnregisterPriority(_autoSContext->_modelPriority, context.deviceInfo.uniqueName);
    // remove the current device from deviceList
    auto eraseDevice = std::find_if(deviceList.begin(), deviceList.end(),
                                    [device](DeviceInformation & d) {
                                        return d.deviceName == device;
                                    });
    deviceList.erase(eraseDevice);
    if (deviceList.empty()) {
        return;
    }
    // select next candidate device
    try {
        std::lock_guard<std::mutex> lock(_autoSContext->_confMutex);
        context.deviceInfo = _autoSContext->_plugin->SelectDevice(deviceList,
                context.networkPrecision, _autoSContext->_modelPriority);
    } catch (const std::exception& e) {
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
    LOG_DEBUG_TAG("try to load %s", context.deviceInfo.deviceName.c_str());
    // try to load this candidate device
    TryToLoadNetWork(context, modelPath, network);
}

void AutoSchedule::WaitFirstNetworkReady() {
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
            LOG_ERROR_TAG("load failed, %s", _loadContext[i].errMessage.c_str());
        }
    }
    IE_THROW() << GetLogTag() << "load all devices failed";
}

void AutoSchedule::WaitActualNetworkReady() const {
    OV_ITT_SCOPED_TASK(itt::domains::MULTIPlugin, "AutoSchedule::WaitActualNetworkReady");
    // Maybe different API will call this function, so add call once here
    // for every AutoSchedule instance
    std::call_once(_oc, [this]() {
        if (_loadContext[ACTUALDEVICE].future.valid()) {
            _loadContext[ACTUALDEVICE].future.wait();
        }
    });
}

bool AutoSchedule::ScheduleToWorkerInferRequest(IE::Task inferPipelineTask, DeviceName preferred_device) {
    std::vector<DeviceInformation> devices;
    // AUTO work mode
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
    for (auto&& device : devices) {
        if (!preferred_device.empty() && (device.deviceName != preferred_device)) {
            continue;
        }
        if (RunPipelineTask(inferPipelineTask, _idleWorkerRequests[device.deviceName], preferred_device)) {
            return true;
        }
    }
    // no vacant requests this time, storing the task to the respective queue
    if (!preferred_device.empty()) {
        _inferPipelineTasksDeviceSpecific[preferred_device]->push(std::move(inferPipelineTask));
    } else {
        _inferPipelineTasks.push(std::move(inferPipelineTask));
    }
    return false;
}

bool AutoSchedule::RunPipelineTask(IE::Task& inferPipelineTask,
    NotBusyPriorityWorkerRequests& idleWorkerRequests,
    const DeviceName& preferred_device) {
    WorkerInferRequest* workerRequestPtr = nullptr;
    std::pair<int, WorkerInferRequest*> worker;
    if (idleWorkerRequests.try_pop(worker)) {
        workerRequestPtr = worker.second;
        IdleGuard<NotBusyPriorityWorkerRequests> idleGuard{workerRequestPtr, idleWorkerRequests};
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

AutoSchedule::~AutoSchedule() {
    // this is necessary to guarantee member destroyed after getting future
    if (_loadContext[CPU].isEnabled) {
        _exitFlag = true;
        _loadContext[CPU].future.wait();
        WaitActualNetworkReady();
        // it's necessary to wait the loading network threads to stop here.
        _autoSContext->_plugin->executorManager()->clear("AutoDeviceAsyncLoad");
        _executor.reset();
    }
    _autoSContext->_plugin->UnregisterPriority(_autoSContext->_modelPriority,
        _loadContext[ACTUALDEVICE].deviceInfo.uniqueName);

    LOG_INFO_TAG("ExecutableNetwork end");
}

IInferPtr AutoSchedule::CreateInferRequest() {
    auto execNetwork = std::dynamic_pointer_cast<AutoExecutableNetwork>(
            _autoSContext->_executableNetwork.lock());
    IInferPtr syncRequestImpl;
    if (_multiSContext->_core && _multiSContext->_core->isNewAPI())
        syncRequestImpl = CreateInferRequestImpl(execNetwork->_parameters, execNetwork->_results);
    if (!syncRequestImpl)
        syncRequestImpl = CreateInferRequestImpl(execNetwork->_networkInputs, execNetwork->_networkOutputs);
    syncRequestImpl->setPointerToExecutableNetworkInternal(execNetwork);
    bool isCumulative = (_autoSContext->_performanceHint == IE::PluginConfigParams::CUMULATIVE_THROUGHPUT) ? true : false;
    if (_passthroughExeNet && !isCumulative) {
        std::string perfmode;
        try {
            perfmode = _passthroughExeNet->GetConfig(
                                CONFIG_KEY(PERFORMANCE_HINT)).as<std::string>();
        } catch(...) {
            LOG_INFO("query perf hint from passthrough network failed");
        }
        if (_autoSContext->_batchingDisabled || perfmode != CONFIG_VALUE(THROUGHPUT)) {
            syncRequestImpl->setPointerToSo(_passthroughExeNet._so);
        } else {
            auto so = _passthroughExeNet._ptr->GetPointerToSo();
            // Get the _so from passthrough executable network when batch plugin is disable.
            if (!so)
                so = _passthroughExeNet._so;
            syncRequestImpl->setPointerToSo(so);
        }
    } else if (std::static_pointer_cast<MultiDeviceInferRequest>(syncRequestImpl)->GetSharedRequest()) {
        // cumulative case, load to MULTI:*
        auto sharedMultiRequest = std::static_pointer_cast<MultiDeviceInferRequest>(syncRequestImpl)->GetSharedRequest();
        if (sharedMultiRequest._ptr->getPointerToSo())
            syncRequestImpl->setPointerToSo(sharedMultiRequest._ptr->getPointerToSo());
        else
            syncRequestImpl->setPointerToSo(sharedMultiRequest._so);
    }
    return std::make_shared<AsyncInferRequest>(shared_from_this(),
                                               syncRequestImpl,
                                               execNetwork->_callbackExecutor);
}
}  // namespace MultiDevicePlugin
