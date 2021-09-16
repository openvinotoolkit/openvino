// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///////////////////////////////////////////////////////////////////////////////////////////////////
#pragma once

#include <atomic>
#include <mutex>
#include <queue>
#include <unordered_map>
#include <map>
#include <vector>
#include <utility>
#include <memory>
#include <string>

#include <cpp_interfaces/impl/ie_executable_network_thread_safe_default.hpp>
#include <cpp_interfaces/impl/ie_infer_async_request_thread_safe_default.hpp>
#include <cpp_interfaces/interface/ie_iplugin_internal.hpp>

#include <ie_parallel.hpp>
#if (IE_THREAD == IE_THREAD_TBB || IE_THREAD == IE_THREAD_TBB_AUTO)
# include <tbb/concurrent_queue.h>
#endif

namespace AutoBatchPlugin {

using DeviceName = std::string;

struct DeviceInformation {
    DeviceName deviceName;
    std::map<std::string, std::string> config;
    int batchForDevice;
};

#if ((IE_THREAD == IE_THREAD_TBB) || (IE_THREAD == IE_THREAD_TBB_AUTO))
template <typename T>
using ThreadSafeQueue = tbb::concurrent_queue<T>;
#else
template <typename T>
class ThreadSafeQueue {
public:
    void push(T value) {
        std::lock_guard<std::mutex> lock(_mutex);
        _queue.push(std::move(value));
    }

    bool try_pop(T& value) {
        std::lock_guard<std::mutex> lock(_mutex);
        if (!_queue.empty()) {
            value = std::move(_queue.front());
            _queue.pop();
            return true;
        } else {
            return false;
        }
    }

    bool empty() {
        std::lock_guard<std::mutex> lock(_mutex);
        return _queue.empty();
    }

protected:
    std::queue<T>   _queue;
    std::mutex      _mutex;
};
#endif

class AutoBatchAsyncInferRequest;
class AutoBatchExecutableNetwork : public InferenceEngine::ExecutableNetworkThreadSafeDefault {
public:
    using Ptr = std::shared_ptr<AutoBatchExecutableNetwork>;
    struct WorkerInferRequest {
        using Ptr = std::shared_ptr<WorkerInferRequest>;
        InferenceEngine::SoIInferRequestInternal   _inferRequest;
        InferenceEngine::StatusCode     _status = InferenceEngine::StatusCode::OK;
        int                             _batchSize;
        std::promise<void>              _cond;
        std::shared_future<void>        _event;
        std::atomic_int                 _numRequestsReady = {0};
    };
    using NotBusyWorkerRequests = ThreadSafeQueue<WorkerInferRequest*>;

    explicit AutoBatchExecutableNetwork(const InferenceEngine::SoExecutableNetworkInternal& networkForDevice,
                                          const DeviceInformation&                                 networkDevices,
                                          const std::unordered_map<std::string, InferenceEngine::Parameter>&    config,
                                          const bool                                                            needPerfCounters = false);

    void SetConfig(const std::map<std::string, InferenceEngine::Parameter> &config) override;
    InferenceEngine::Parameter GetConfig(const std::string &name) const override;
    InferenceEngine::Parameter GetMetric(const std::string &name) const override;
    InferenceEngine::IInferRequestInternal::Ptr CreateInferRequest() override;
    InferenceEngine::IInferRequestInternal::Ptr CreateInferRequestImpl(InferenceEngine::InputsDataMap networkInputs,
                                                                      InferenceEngine::OutputsDataMap networkOutputs) override;
    virtual ~AutoBatchExecutableNetwork();

    std::atomic_bool                                            _terminate = {false};
    DeviceInformation                                           _device;
    InferenceEngine::SoExecutableNetworkInternal                _network;
    std::vector<WorkerInferRequest::Ptr>                        _workerRequests;
    std::unordered_map<std::string, InferenceEngine::Parameter> _config;
    bool                                                        _needPerfCounters = false;
    std::atomic_size_t                                          _numRequestsCreated = {0};
};

class AutoBatchInferRequest : public InferenceEngine::IInferRequestInternal {
public:
    using Ptr = std::shared_ptr<AutoBatchInferRequest>;
    explicit AutoBatchInferRequest(const InferenceEngine::InputsDataMap&  networkInputs,
                                   const InferenceEngine::OutputsDataMap& networkOutputs,
                                   AutoBatchExecutableNetwork::WorkerInferRequest* workerRequestPtr,
                                   int batch_id, int num_batch, bool _needPerfCounters = false);
    std::map<std::string, InferenceEngine::InferenceEngineProfileInfo> GetPerformanceCounts() const override;
    void InferImpl() override;

    // Batch-Device impl specific: sets the data (blobs from the device request to the batched device request)
    void SetBlobsToAnotherRequest(InferenceEngine::InferRequest& req);
    AutoBatchExecutableNetwork::WorkerInferRequest* _workerInferRequest;
protected:
    std::map<std::string, InferenceEngine::InferenceEngineProfileInfo>  _perfMap;
    bool                                                                _needPerfCounters = false;
};

class AutoBatchAsyncInferRequest : public InferenceEngine::AsyncInferRequestThreadSafeDefault {
public:
    using Ptr = std::shared_ptr<AutoBatchAsyncInferRequest>;

    explicit AutoBatchAsyncInferRequest(const AutoBatchInferRequest::Ptr&           inferRequest,
                                          const bool                                needPerfCounters,
                                          const AutoBatchExecutableNetwork::Ptr&      AutoBatchExecutableNetwork,
                                          const InferenceEngine::ITaskExecutor::Ptr&    callbackExecutor);
    void Infer_ThreadUnsafe() override;
    virtual ~AutoBatchAsyncInferRequest();

protected:
    AutoBatchExecutableNetwork::Ptr                                   _AutoBatchExecutableNetwork;
    AutoBatchInferRequest::Ptr                                        _inferRequest;
};

class AutoBatchInferencePlugin : public InferenceEngine::IInferencePlugin {
public:
    AutoBatchInferencePlugin();
    virtual ~AutoBatchInferencePlugin() = default;

    InferenceEngine::IExecutableNetworkInternal::Ptr LoadExeNetworkImpl(const InferenceEngine::CNNNetwork& network,
                                                                       const std::map<std::string, std::string>& config) override;

    void SetConfig(const std::map<std::string, std::string>& config) override;
    InferenceEngine::Parameter GetConfig(const std::string& name,
                        const std::map<std::string, InferenceEngine::Parameter> & options) const override;
    InferenceEngine::QueryNetworkResult QueryNetwork(const InferenceEngine::CNNNetwork&       network,
                      const std::map<std::string, std::string>& config) const override;
    InferenceEngine::Parameter GetMetric(const std::string& name,
                                         const std::map<std::string, InferenceEngine::Parameter>& options) const override;

    DeviceInformation ParseMetaDevice(const std::string & devicesBatchCfg,
                                                  const std::map<std::string, std::string> & config) const;

protected:
    std::map<std::string, std::string> GetSupportedConfig(const std::map<std::string, std::string>& config,
                                                          const DeviceName & deviceName) const;
};

}  // namespace AutoBatchPlugin
