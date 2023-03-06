// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///////////////////////////////////////////////////////////////////////////////////////////////////
#include "auto_batch.hpp"

#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "cpp_interfaces/interface/ie_internal_plugin_config.hpp"
#include "dimension_tracker.hpp"
#include "ie_icore.hpp"
#include "ie_ngraph_utils.hpp"
#include "ie_performance_hints.hpp"
#include "openvino/pass/manager.hpp"
#include "openvino/runtime/intel_gpu/properties.hpp"
#include "transformations/common_optimizations/dimension_tracking.hpp"
#include "transformations/init_node_info.hpp"
#include "transformations/utils/utils.hpp"

namespace AutoBatchPlugin {
using namespace InferenceEngine;

std::vector<std::string> supported_configKeys = {CONFIG_KEY(AUTO_BATCH_DEVICE_CONFIG),
                                                 CONFIG_KEY(AUTO_BATCH_TIMEOUT),
                                                 CONFIG_KEY(CACHE_DIR)};

template <Precision::ePrecision precision>
Blob::Ptr create_shared_blob_on_top_of_batched_blob(Blob::Ptr batched_blob,
                                                    std::string name,
                                                    const std::set<std::string>& batched_names,
                                                    size_t batch_id,
                                                    size_t batch_num) {
    typedef typename PrecisionTrait<precision>::value_type TYPE;
    typedef typename std::add_pointer<TYPE>::type TYPEPTR;
    auto ptr = batched_blob->buffer().as<TYPEPTR>();
    auto sizePerBatch = batched_blob->size() / batch_num;
    SizeVector dims = batched_blob->getTensorDesc().getDims();
    // for performance reason (copy avoidance) current impl of the auto-batching supports only batching by 0th dim
    if (batched_names.count(name)) {
        dims[0] = 1;
        return make_shared_blob<TYPE>({precision, dims, batched_blob->getTensorDesc().getLayout()},
                                      ptr + sizePerBatch * batch_id,
                                      sizePerBatch);
    } else {
        // same blob for all requests (e.g. constants)
        return make_shared_blob<TYPE>({precision, dims, batched_blob->getTensorDesc().getLayout()}, ptr);
    }
}

// ------------------------------AutoBatchInferRequest----------------------------
AutoBatchInferRequest::AutoBatchInferRequest(const std::vector<std::shared_ptr<const ov::Node>>& inputs,
                                             const std::vector<std::shared_ptr<const ov::Node>>& outputs,
                                             AutoBatchExecutableNetwork::WorkerInferRequest& workerRequest,
                                             int batch_id,
                                             int num_batch,
                                             const std::set<std::string>& batchedInputs,
                                             const std::set<std::string>& batchedOutputs)
    : IInferRequestInternal(inputs, outputs),
      _myBatchedRequestWrapper(workerRequest),
      _batchId(batch_id),
      _batchSize(num_batch) {
    ShareBlobsWithBatchRequest(batchedInputs, batchedOutputs);
}

AutoBatchInferRequest::AutoBatchInferRequest(const InputsDataMap& networkInputs,
                                             const OutputsDataMap& networkOutputs,
                                             AutoBatchExecutableNetwork::WorkerInferRequest& workerRequest,
                                             int batch_id,
                                             int num_batch,
                                             const std::set<std::string>& batchedInputs,
                                             const std::set<std::string>& batchedOutputs)
    : IInferRequestInternal(networkInputs, networkOutputs),
      _myBatchedRequestWrapper(workerRequest),
      _batchId(batch_id),
      _batchSize(num_batch) {
    ShareBlobsWithBatchRequest(batchedInputs, batchedOutputs);
}

void AutoBatchInferRequest::ShareBlobsWithBatchRequest(const std::set<std::string>& batchedInputs,
                                                       const std::set<std::string>& batchedOutputs) {
    // Allocate all input blobs
    for (const auto& it : _networkInputs) {
        auto blob = _myBatchedRequestWrapper._inferRequestBatched->GetBlob(it.first);
        Blob::Ptr res;
        switch (it.second->getTensorDesc().getPrecision()) {
        case InferenceEngine::Precision::FP32:
            res = create_shared_blob_on_top_of_batched_blob<InferenceEngine::Precision::FP32>(
                _myBatchedRequestWrapper._inferRequestBatched->GetBlob(it.first),
                it.first,
                batchedInputs,
                _batchId,
                _batchSize);
            break;
        case InferenceEngine::Precision::I32:
            res = create_shared_blob_on_top_of_batched_blob<InferenceEngine::Precision::I32>(
                _myBatchedRequestWrapper._inferRequestBatched->GetBlob(it.first),
                it.first,
                batchedInputs,
                _batchId,
                _batchSize);
            break;
        case InferenceEngine::Precision::I8:
            res = create_shared_blob_on_top_of_batched_blob<InferenceEngine::Precision::I8>(
                _myBatchedRequestWrapper._inferRequestBatched->GetBlob(it.first),
                it.first,
                batchedInputs,
                _batchId,
                _batchSize);
            break;
        case InferenceEngine::Precision::I16:
            res = create_shared_blob_on_top_of_batched_blob<InferenceEngine::Precision::I16>(
                _myBatchedRequestWrapper._inferRequestBatched->GetBlob(it.first),
                it.first,
                batchedInputs,
                _batchId,
                _batchSize);
            break;
        case InferenceEngine::Precision::U16:
            res = create_shared_blob_on_top_of_batched_blob<InferenceEngine::Precision::U16>(
                _myBatchedRequestWrapper._inferRequestBatched->GetBlob(it.first),
                it.first,
                batchedInputs,
                _batchId,
                _batchSize);
            break;
        case InferenceEngine::Precision::U32:
            res = create_shared_blob_on_top_of_batched_blob<InferenceEngine::Precision::U32>(
                _myBatchedRequestWrapper._inferRequestBatched->GetBlob(it.first),
                it.first,
                batchedInputs,
                _batchId,
                _batchSize);
            break;
        case InferenceEngine::Precision::FP64:
            res = create_shared_blob_on_top_of_batched_blob<InferenceEngine::Precision::FP64>(
                _myBatchedRequestWrapper._inferRequestBatched->GetBlob(it.first),
                it.first,
                batchedInputs,
                _batchId,
                _batchSize);
            break;
        case InferenceEngine::Precision::FP16:
            res = create_shared_blob_on_top_of_batched_blob<InferenceEngine::Precision::FP16>(
                _myBatchedRequestWrapper._inferRequestBatched->GetBlob(it.first),
                it.first,
                batchedInputs,
                _batchId,
                _batchSize);
            break;
        case InferenceEngine::Precision::BF16:
            res = create_shared_blob_on_top_of_batched_blob<InferenceEngine::Precision::BF16>(
                _myBatchedRequestWrapper._inferRequestBatched->GetBlob(it.first),
                it.first,
                batchedInputs,
                _batchId,
                _batchSize);
            break;
        case InferenceEngine::Precision::U64:
            res = create_shared_blob_on_top_of_batched_blob<InferenceEngine::Precision::U64>(
                _myBatchedRequestWrapper._inferRequestBatched->GetBlob(it.first),
                it.first,
                batchedInputs,
                _batchId,
                _batchSize);
            break;
        case InferenceEngine::Precision::I64:
            res = create_shared_blob_on_top_of_batched_blob<InferenceEngine::Precision::I64>(
                _myBatchedRequestWrapper._inferRequestBatched->GetBlob(it.first),
                it.first,
                batchedInputs,
                _batchId,
                _batchSize);
            break;
        case InferenceEngine::Precision::U8:
            res = create_shared_blob_on_top_of_batched_blob<InferenceEngine::Precision::U8>(
                _myBatchedRequestWrapper._inferRequestBatched->GetBlob(it.first),
                it.first,
                batchedInputs,
                _batchId,
                _batchSize);
            break;
        case InferenceEngine::Precision::BOOL:
            res = create_shared_blob_on_top_of_batched_blob<InferenceEngine::Precision::BOOL>(
                _myBatchedRequestWrapper._inferRequestBatched->GetBlob(it.first),
                it.first,
                batchedInputs,
                _batchId,
                _batchSize);
            break;
        default:
            IE_THROW() << "Unsupported input precision " << it.second->getTensorDesc().getPrecision();
        }
        _inputs[it.first] = res;
    }
    // Allocate all output blobs
    for (const auto& it : _networkOutputs) {
        auto blob = _myBatchedRequestWrapper._inferRequestBatched->GetBlob(it.first);
        Blob::Ptr res;
        switch (it.second->getTensorDesc().getPrecision()) {
        case InferenceEngine::Precision::FP32:
            res = create_shared_blob_on_top_of_batched_blob<InferenceEngine::Precision::FP32>(
                _myBatchedRequestWrapper._inferRequestBatched->GetBlob(it.first),
                it.first,
                batchedOutputs,
                _batchId,
                _batchSize);
            break;
        case InferenceEngine::Precision::I32:
            res = create_shared_blob_on_top_of_batched_blob<InferenceEngine::Precision::I32>(
                _myBatchedRequestWrapper._inferRequestBatched->GetBlob(it.first),
                it.first,
                batchedOutputs,
                _batchId,
                _batchSize);
            break;
        case InferenceEngine::Precision::I8:
            res = create_shared_blob_on_top_of_batched_blob<InferenceEngine::Precision::I8>(
                _myBatchedRequestWrapper._inferRequestBatched->GetBlob(it.first),
                it.first,
                batchedOutputs,
                _batchId,
                _batchSize);
            break;
        case InferenceEngine::Precision::I16:
            res = create_shared_blob_on_top_of_batched_blob<InferenceEngine::Precision::I16>(
                _myBatchedRequestWrapper._inferRequestBatched->GetBlob(it.first),
                it.first,
                batchedOutputs,
                _batchId,
                _batchSize);
            break;
        case InferenceEngine::Precision::U16:
            res = create_shared_blob_on_top_of_batched_blob<InferenceEngine::Precision::U16>(
                _myBatchedRequestWrapper._inferRequestBatched->GetBlob(it.first),
                it.first,
                batchedOutputs,
                _batchId,
                _batchSize);
            break;
        case InferenceEngine::Precision::U32:
            res = create_shared_blob_on_top_of_batched_blob<InferenceEngine::Precision::U32>(
                _myBatchedRequestWrapper._inferRequestBatched->GetBlob(it.first),
                it.first,
                batchedOutputs,
                _batchId,
                _batchSize);
            break;
        case InferenceEngine::Precision::FP64:
            res = create_shared_blob_on_top_of_batched_blob<InferenceEngine::Precision::FP64>(
                _myBatchedRequestWrapper._inferRequestBatched->GetBlob(it.first),
                it.first,
                batchedOutputs,
                _batchId,
                _batchSize);
            break;
        case InferenceEngine::Precision::FP16:
            res = create_shared_blob_on_top_of_batched_blob<InferenceEngine::Precision::FP16>(
                _myBatchedRequestWrapper._inferRequestBatched->GetBlob(it.first),
                it.first,
                batchedOutputs,
                _batchId,
                _batchSize);
            break;
        case InferenceEngine::Precision::BF16:
            res = create_shared_blob_on_top_of_batched_blob<InferenceEngine::Precision::BF16>(
                _myBatchedRequestWrapper._inferRequestBatched->GetBlob(it.first),
                it.first,
                batchedOutputs,
                _batchId,
                _batchSize);
            break;
        case InferenceEngine::Precision::U64:
            res = create_shared_blob_on_top_of_batched_blob<InferenceEngine::Precision::U64>(
                _myBatchedRequestWrapper._inferRequestBatched->GetBlob(it.first),
                it.first,
                batchedOutputs,
                _batchId,
                _batchSize);
            break;
        case InferenceEngine::Precision::I64:
            res = create_shared_blob_on_top_of_batched_blob<InferenceEngine::Precision::I64>(
                _myBatchedRequestWrapper._inferRequestBatched->GetBlob(it.first),
                it.first,
                batchedOutputs,
                _batchId,
                _batchSize);
            break;
        case InferenceEngine::Precision::U8:
            res = create_shared_blob_on_top_of_batched_blob<InferenceEngine::Precision::U8>(
                _myBatchedRequestWrapper._inferRequestBatched->GetBlob(it.first),
                it.first,
                batchedOutputs,
                _batchId,
                _batchSize);
            break;
        case InferenceEngine::Precision::BOOL:
            res = create_shared_blob_on_top_of_batched_blob<InferenceEngine::Precision::BOOL>(
                _myBatchedRequestWrapper._inferRequestBatched->GetBlob(it.first),
                it.first,
                batchedOutputs,
                _batchId,
                _batchSize);
            break;
        default:
            IE_THROW(NotImplemented) << "Unsupported input precision " << it.second->getTensorDesc().getPrecision();
        }
        _outputs[it.first] = res;
    }
}
void AutoBatchInferRequest::SetBlobsToAnotherRequest(SoIInferRequestInternal& req) {
    for (const auto& it : _networkInputs) {
        auto& name = it.first;
        // this request is already in BUSY state, so using the internal functions safely
        auto blob = GetBlob(name);
        if (req->GetBlob(name) != blob)
            req->SetBlob(name, blob);
    }
    for (const auto& it : _networkOutputs) {
        auto& name = it.first;
        // this request is already in BUSY state, so using the internal functions safely
        auto blob = GetBlob(name);
        if (req->GetBlob(name) != blob)
            req->SetBlob(name, blob);
    }
}

void AutoBatchInferRequest::CopyInputsIfNeeded() {
    for (const auto& it : _networkInputs) {
        auto& name = it.first;
        // this request is already in BUSY state, so using the internal functions safely
        CopyBlobIfNeeded(GetBlob(name), _myBatchedRequestWrapper._inferRequestBatched->GetBlob(name), true);
    }
}

void AutoBatchInferRequest::CopyBlobIfNeeded(InferenceEngine::Blob::CPtr src,
                                             InferenceEngine::Blob::Ptr dst,
                                             bool bInput) {
    auto bufferDst = dst->buffer();
    auto ptrDst = bufferDst.as<char*>();
    auto bufferSrc = src->cbuffer();
    auto ptrSrc = bufferSrc.as<const char*>();
    ptrdiff_t szDst = dst->byteSize();
    ptrdiff_t szSrc = src->byteSize();
    if (bInput) {
        ptrdiff_t offset = szSrc != szDst ? _batchId * szDst / _batchSize : 0;
        if ((ptrDst + offset) == ptrSrc)
            return;
        else
            memcpy(ptrDst + offset, ptrSrc, szSrc);
    } else {
        ptrdiff_t offset = szSrc != szDst ? _batchId * szSrc / _batchSize : 0;
        if ((ptrSrc + offset) == ptrDst)
            return;
        else
            memcpy(ptrDst, ptrSrc + offset, szDst);
    }
}

void AutoBatchInferRequest::CopyOutputsIfNeeded() {
    for (const auto& it : _networkOutputs) {
        auto& name = it.first;
        // this request is already in BUSY state, so using the internal functions safely
        CopyBlobIfNeeded(_myBatchedRequestWrapper._inferRequestBatched->GetBlob(name), GetBlob(name), false);
    }
}

AutoBatchAsyncInferRequest::AutoBatchAsyncInferRequest(
    const AutoBatchInferRequest::Ptr& inferRequest,
    InferenceEngine::SoIInferRequestInternal& inferRequestWithoutBatch,
    const ITaskExecutor::Ptr& callbackExecutor)
    : AsyncInferRequestThreadSafeDefault(inferRequest, nullptr, callbackExecutor),
      _inferRequestWithoutBatch(inferRequestWithoutBatch),
      _inferRequest{inferRequest} {
    // this executor starts the inference while  the task (checking the result) is passed to the next stage
    struct ThisRequestExecutor : public ITaskExecutor {
        explicit ThisRequestExecutor(AutoBatchAsyncInferRequest* _this_) : _this{_this_} {}
        void run(Task task) override {
            auto& workerInferRequest = _this->_inferRequest->_myBatchedRequestWrapper;
            std::pair<AutoBatchAsyncInferRequest*, InferenceEngine::Task> t;
            t.first = _this;
            t.second = std::move(task);
            workerInferRequest._tasks.push(t);
            // it is ok to call size() here as the queue only grows (and the bulk removal happens under the mutex)
            const int sz = static_cast<int>(workerInferRequest._tasks.size());
            if (sz == workerInferRequest._batchSize) {
                workerInferRequest._cond.notify_one();
            }
        };
        AutoBatchAsyncInferRequest* _this = nullptr;
    };
    _pipeline = {{/*TaskExecutor*/ std::make_shared<ThisRequestExecutor>(this), /*task*/ [this] {
                      if (this->_inferRequest->_exceptionPtr)  // if the exception happened in the batch1 fallback
                          std::rethrow_exception(this->_inferRequest->_exceptionPtr);
                      auto& batchReq = this->_inferRequest->_myBatchedRequestWrapper;
                      if (batchReq._exceptionPtr)  // when the batchN execution failed
                          std::rethrow_exception(batchReq._exceptionPtr);
                      // in the case of non-batched execution the blobs were set explicitly
                      if (AutoBatchInferRequest::eExecutionFlavor::BATCH_EXECUTED ==
                          this->_inferRequest->_wasBatchedRequestUsed)
                          this->_inferRequest->CopyOutputsIfNeeded();
                  }}};
}

std::map<std::string, InferenceEngine::InferenceEngineProfileInfo> AutoBatchAsyncInferRequest::GetPerformanceCounts()
    const {
    CheckState();
    if (AutoBatchInferRequest::eExecutionFlavor::BATCH_EXECUTED == _inferRequest->_wasBatchedRequestUsed)
        return _inferRequest->_myBatchedRequestWrapper._inferRequestBatched->GetPerformanceCounts();
    else
        return _inferRequestWithoutBatch->GetPerformanceCounts();
}

void AutoBatchAsyncInferRequest::Infer_ThreadUnsafe() {
    InferUsingAsync();
}

AutoBatchAsyncInferRequest::~AutoBatchAsyncInferRequest() {
    StopAndWait();
}

// ------------------------------AutoBatchExecutableNetwork----------------------------
AutoBatchExecutableNetwork::AutoBatchExecutableNetwork(
    const InferenceEngine::SoExecutableNetworkInternal& networkWithBatch,
    const InferenceEngine::SoExecutableNetworkInternal& networkWithoutBatch,
    const DeviceInformation& networkDevice,
    const std::unordered_map<std::string, InferenceEngine::Parameter>& config,
    const std::set<std::string>& batchedInputs,
    const std::set<std::string>& batchedOutputs)
    : InferenceEngine::ExecutableNetworkThreadSafeDefault(nullptr,
                                                          std::make_shared<InferenceEngine::ImmediateExecutor>()),
      _network{networkWithBatch},
      _networkWithoutBatch{networkWithoutBatch},
      _config{config},
      _batchedInputs(batchedInputs),
      _batchedOutputs(batchedOutputs) {
    // WA for gcc 4.8 ( fails compilation with member init-list)
    _device = networkDevice;
    auto time_out = config.find(CONFIG_KEY(AUTO_BATCH_TIMEOUT));
    IE_ASSERT(time_out != config.end());
    _timeOut = ParseTimeoutValue(time_out->second.as<std::string>());
}

AutoBatchExecutableNetwork::~AutoBatchExecutableNetwork() {
    _terminate = true;
    for (auto w : _workerRequests) {
        w->_thread.join();
    }
    _workerRequests.clear();
}

unsigned int AutoBatchExecutableNetwork::ParseTimeoutValue(const std::string& s) {
    auto val = std::stoi(s);
    if (val < 0)
        IE_THROW(ParameterMismatch) << "Value for the " << CONFIG_KEY(AUTO_BATCH_TIMEOUT) << " should be unsigned int";
    return val;
}

std::shared_ptr<InferenceEngine::RemoteContext> AutoBatchExecutableNetwork::GetContext() const {
    return _networkWithoutBatch->GetContext();
}

InferenceEngine::IInferRequestInternal::Ptr AutoBatchExecutableNetwork::CreateInferRequestImpl(
    InferenceEngine::InputsDataMap networkInputs,
    InferenceEngine::OutputsDataMap networkOutputs) {
    auto workerRequestPtrAndId = GetWorkerInferRequest();
    return std::make_shared<AutoBatchInferRequest>(networkInputs,
                                                   networkOutputs,
                                                   workerRequestPtrAndId.first,
                                                   workerRequestPtrAndId.second,
                                                   _device.batchForDevice,
                                                   _batchedInputs,
                                                   _batchedOutputs);
}

InferenceEngine::IInferRequestInternal::Ptr AutoBatchExecutableNetwork::CreateInferRequestImpl(
    const std::vector<std::shared_ptr<const ov::Node>>& inputs,
    const std::vector<std::shared_ptr<const ov::Node>>& outputs) {
    if (!this->_plugin || !_plugin->IsNewAPI())
        return nullptr;
    auto workerRequestPtrAndId = GetWorkerInferRequest();
    return std::make_shared<AutoBatchInferRequest>(inputs,
                                                   outputs,
                                                   workerRequestPtrAndId.first,
                                                   workerRequestPtrAndId.second,
                                                   _device.batchForDevice,
                                                   _batchedInputs,
                                                   _batchedOutputs);
}

std::pair<AutoBatchExecutableNetwork::WorkerInferRequest&, int> AutoBatchExecutableNetwork::GetWorkerInferRequest() {
    auto num = _numRequestsCreated++;
    std::lock_guard<std::mutex> lock(_workerRequestsMutex);
    auto batch_id = num % _device.batchForDevice;
    if (!batch_id) {  // need new request
        _workerRequests.push_back(std::make_shared<WorkerInferRequest>());
        auto workerRequestPtr = _workerRequests.back().get();
        workerRequestPtr->_inferRequestBatched = {_network->CreateInferRequest(), _network._so};
        workerRequestPtr->_batchSize = _device.batchForDevice;
        workerRequestPtr->_completionTasks.resize(workerRequestPtr->_batchSize);
        workerRequestPtr->_inferRequestBatched->SetCallback(
            [workerRequestPtr](std::exception_ptr exceptionPtr) mutable {
                if (exceptionPtr)
                    workerRequestPtr->_exceptionPtr = exceptionPtr;
                IE_ASSERT(workerRequestPtr->_completionTasks.size() == (size_t)workerRequestPtr->_batchSize);
                // notify the individual requests on the completion
                for (int c = 0; c < workerRequestPtr->_batchSize; c++) {
                    workerRequestPtr->_completionTasks[c]();
                }
                // reset the timeout
                workerRequestPtr->_cond.notify_one();
            });

        workerRequestPtr->_thread = std::thread([workerRequestPtr, this] {
            while (1) {
                std::cv_status status;
                {
                    std::unique_lock<std::mutex> lock(workerRequestPtr->_mutex);
                    status = workerRequestPtr->_cond.wait_for(lock, std::chrono::milliseconds(_timeOut));
                }
                if (_terminate) {
                    break;
                } else {
                    // as we pop the tasks from the queue only here
                    // it is ok to call size() (as the _tasks can only grow in parallel)
                    const int sz = static_cast<int>(workerRequestPtr->_tasks.size());
                    if (sz == workerRequestPtr->_batchSize) {
                        std::pair<AutoBatchAsyncInferRequest*, InferenceEngine::Task> t;
                        for (int n = 0; n < sz; n++) {
                            IE_ASSERT(workerRequestPtr->_tasks.try_pop(t));
                            workerRequestPtr->_completionTasks[n] = std::move(t.second);
                            t.first->_inferRequest->CopyInputsIfNeeded();
                            t.first->_inferRequest->_wasBatchedRequestUsed =
                                AutoBatchInferRequest::eExecutionFlavor::BATCH_EXECUTED;
                        }
                        workerRequestPtr->_inferRequestBatched->StartAsync();
                    } else if ((status == std::cv_status::timeout) && sz) {
                        // timeout to collect the batch is over, have to execute the requests in the batch1 mode
                        std::pair<AutoBatchAsyncInferRequest*, InferenceEngine::Task> t;
                        // popping all tasks collected by the moment of the time-out and execute each with batch1
                        std::atomic<int> arrived = {0};
                        std::promise<void> all_completed;
                        auto all_completed_future = all_completed.get_future();
                        for (int n = 0; n < sz; n++) {
                            IE_ASSERT(workerRequestPtr->_tasks.try_pop(t));
                            t.first->_inferRequestWithoutBatch->SetCallback(
                                [t, sz, &arrived, &all_completed](std::exception_ptr p) {
                                    if (p)
                                        t.first->_inferRequest->_exceptionPtr = p;
                                    t.second();
                                    if (sz == ++arrived)
                                        all_completed.set_value();
                                });
                            t.first->_inferRequest->_wasBatchedRequestUsed =
                                AutoBatchInferRequest::eExecutionFlavor::TIMEOUT_EXECUTED;
                            t.first->_inferRequest->SetBlobsToAnotherRequest(t.first->_inferRequestWithoutBatch);
                            t.first->_inferRequestWithoutBatch->StartAsync();
                        }
                        all_completed_future.get();
                        // now when all the tasks for this batch are completed, start waiting for the timeout again
                    }
                }
            }
        });
    }
    return {*_workerRequests.back(), static_cast<int>(batch_id)};
}

InferenceEngine::IInferRequestInternal::Ptr AutoBatchExecutableNetwork::CreateInferRequest() {
    if (!_network) {
        auto res = _networkWithoutBatch->CreateInferRequest();
        res->setPointerToExecutableNetworkInternal(shared_from_this());
        res->setPointerToSo(_networkWithoutBatch._so);
        _so = _networkWithoutBatch._so;
        return res;
    }
    // trying to create the new API request first
    IInferRequestInternal::Ptr syncRequestImpl = CreateInferRequestImpl(_parameters, _results);
    if (!syncRequestImpl)
        syncRequestImpl = CreateInferRequestImpl(_networkInputs, _networkOutputs);
    syncRequestImpl->setPointerToExecutableNetworkInternal(shared_from_this());
    InferenceEngine::SoIInferRequestInternal inferRequestWithoutBatch = {_networkWithoutBatch->CreateInferRequest(),
                                                                         _networkWithoutBatch._so};
    return std::make_shared<AutoBatchAsyncInferRequest>(
        std::static_pointer_cast<AutoBatchInferRequest>(syncRequestImpl),
        inferRequestWithoutBatch,
        _callbackExecutor);
}

std::shared_ptr<ngraph::Function> AutoBatchExecutableNetwork::GetExecGraphInfo() {
    return _network && _network->GetExecGraphInfo() ? _network->GetExecGraphInfo()
                                                    : _networkWithoutBatch->GetExecGraphInfo();
}

void AutoBatchExecutableNetwork::SetConfig(const std::map<std::string, InferenceEngine::Parameter>& config) {
    auto timeout = config.find(CONFIG_KEY(AUTO_BATCH_TIMEOUT));
    if (timeout == config.end() || config.size() > 1) {
        IE_THROW() << "The only config that can be changed on the fly for the AutoBatching the is the "
                   << CONFIG_KEY(AUTO_BATCH_TIMEOUT);
    } else {
        _timeOut = ParseTimeoutValue(timeout->second.as<std::string>());
    }
}

InferenceEngine::Parameter AutoBatchExecutableNetwork::GetConfig(const std::string& name) const {
    auto it = _config.find(name);
    if (it != _config.end()) {
        return it->second;
    } else {
        // find config key among networks config keys
        auto param = _networkWithoutBatch->GetMetric(METRIC_KEY(SUPPORTED_CONFIG_KEYS));
        for (auto&& configKey : param.as<std::vector<std::string>>()) {
            if (configKey == name) {
                return _networkWithoutBatch->GetConfig(configKey);
            }
        }
        IE_THROW(NotFound) << name << " not found in the ExecutableNetwork config";
    }
}

InferenceEngine::Parameter AutoBatchExecutableNetwork::GetMetric(const std::string& name) const {
    if (name == METRIC_KEY(OPTIMAL_NUMBER_OF_INFER_REQUESTS)) {
        auto reqs = 0;
        try {
            auto hint = _networkWithoutBatch->GetConfig(CONFIG_KEY(PERFORMANCE_HINT_NUM_REQUESTS)).as<std::string>();
            reqs = InferenceEngine::PerfHintsConfig::CheckPerformanceHintRequestValue(hint);
            if (!reqs)  // no limitations from user, let's deduce the full blown #requests
                // (multiplied by the devices capabilities to run multiple <batched> requests for further perf)
                reqs = _device.batchForDevice *
                       _networkWithoutBatch->GetMetric(METRIC_KEY(OPTIMAL_NUMBER_OF_INFER_REQUESTS)).as<unsigned int>();
        } catch (const InferenceEngine::Exception&) {
        }
        reqs = std::max(reqs, _device.batchForDevice);  // round up to the possible  user's value
        IE_SET_METRIC_RETURN(OPTIMAL_NUMBER_OF_INFER_REQUESTS, reqs);
    } else if (name == METRIC_KEY(NETWORK_NAME)) {
        IE_SET_METRIC_RETURN(NETWORK_NAME, _networkWithoutBatch->GetMetric(METRIC_KEY(NETWORK_NAME)).as<std::string>());
    } else if (name == METRIC_KEY(SUPPORTED_METRICS)) {
        IE_SET_METRIC_RETURN(SUPPORTED_METRICS,
                             {METRIC_KEY(OPTIMAL_NUMBER_OF_INFER_REQUESTS),
                              METRIC_KEY(SUPPORTED_METRICS),
                              METRIC_KEY(NETWORK_NAME),
                              METRIC_KEY(SUPPORTED_CONFIG_KEYS),
                              ov::execution_devices.name()});
    } else if (name == METRIC_KEY(SUPPORTED_CONFIG_KEYS)) {
        IE_SET_METRIC_RETURN(SUPPORTED_CONFIG_KEYS,
                             {CONFIG_KEY(AUTO_BATCH_TIMEOUT)});  // only timeout can be changed on the fly
    } else if (name == ov::execution_devices) {
        return _networkWithoutBatch->GetMetric(name);
    } else {
        IE_THROW() << "Unsupported Network metric: " << name;
    }
}

// ------------------------------AutoBatchInferencePlugin----------------------------

namespace {

std::map<std::string, std::string> mergeConfigs(std::map<std::string, std::string> config,
                                                const std::map<std::string, std::string>& local) {
    for (auto&& kvp : local) {
        config[kvp.first] = kvp.second;
    }
    return config;
}

}  // namespace

DeviceInformation AutoBatchInferencePlugin::ParseBatchDevice(const std::string& deviceWithBatch) {
    auto&& d = deviceWithBatch;
    auto openingBracket = d.find_first_of('(');
    auto closingBracket = d.find_first_of(')', openingBracket);
    auto deviceName = d.substr(0, openingBracket);

    int batch = 0;
    if (closingBracket != std::string::npos && openingBracket < closingBracket) {
        batch = std::stol(d.substr(openingBracket + 1, closingBracket - 1));

        if (batch <= 0) {
            IE_THROW() << "Batch value for '" << deviceName << "' must be > 0, while " << batch << "is passed";
        }
    }
    return {deviceName, {{}}, batch};
}

DeviceInformation AutoBatchInferencePlugin::ParseMetaDevice(const std::string& devicesBatchCfg,
                                                            const std::map<std::string, std::string>& config) const {
    auto getDeviceConfig = [&](const DeviceName& deviceWithID) {
        DeviceIDParser deviceParser(deviceWithID);
        std::string deviceName = deviceParser.getDeviceName();
        std::map<std::string, std::string> tconfig = mergeConfigs(_config, config);

        // set device ID if any
        std::string deviceIDLocal = deviceParser.getDeviceID();
        if (!deviceIDLocal.empty()) {
            tconfig[PluginConfigParams::KEY_DEVICE_ID] = deviceIDLocal;
        }
        // passthrough the cache dir to core->loadnetwork when underlying device does not support cache dir
        auto deviceConfig = GetCore()->GetSupportedConfig(deviceName, tconfig);
        if (tconfig.find(CONFIG_KEY(CACHE_DIR)) != tconfig.end() &&
            deviceConfig.find(CONFIG_KEY(CACHE_DIR)) == deviceConfig.end()) {
            auto tmpiter = tconfig.find(CONFIG_KEY(CACHE_DIR));
            if (tmpiter != tconfig.end())
                deviceConfig.insert({tmpiter->first, tmpiter->second});
        }
        return deviceConfig;
    };

    auto metaDevice = ParseBatchDevice(devicesBatchCfg);
    metaDevice.config = getDeviceConfig(metaDevice.deviceName);

    auto cfg = config;
    // check that no irrelevant config-keys left
    for (auto k : config) {
        const auto& name = k.first;
        auto found_in_supported_cfg = std::find(supported_configKeys.begin(), supported_configKeys.end(), k.first);
        auto found_in_device_cfg = metaDevice.config.find(k.first);
        if (found_in_device_cfg == metaDevice.config.end() && found_in_supported_cfg == supported_configKeys.end()) {
            IE_THROW() << "Unsupported config key: " << name;
        }
    }
    return metaDevice;
}

RemoteContext::Ptr AutoBatchInferencePlugin::CreateContext(const InferenceEngine::ParamMap& config) {
    auto cfg = config;
    auto it = cfg.find(CONFIG_KEY(AUTO_BATCH_DEVICE_CONFIG));
    if (it == cfg.end())
        IE_THROW() << "Value for KEY_AUTO_BATCH_DEVICE_CONFIG is not set";

    auto val = it->second.as<std::string>();
    auto core = GetCore();
    if (!core)
        return nullptr;
    auto metaDevice = ParseMetaDevice(val, std::map<std::string, std::string>());
    cfg.erase(it);
    return core->CreateContext(metaDevice.deviceName, cfg);
}

Parameter AutoBatchInferencePlugin::GetConfig(const std::string& name,
                                              const std::map<std::string, Parameter>& options) const {
    if (supported_configKeys.end() != std::find(supported_configKeys.begin(), supported_configKeys.end(), name)) {
        auto it = _config.find(name);
        if (it == _config.end()) {
            IE_THROW() << "Value for " << name << " is not set";
        } else {
            return {it->second};
        }
    } else {
        IE_THROW() << "Unsupported config key: " << name;
    }
}

void AutoBatchInferencePlugin::CheckConfig(const std::map<std::string, std::string>& config) {
    for (auto&& kvp : config) {
        const auto name = kvp.first;
        const auto val = kvp.second;
        if (supported_configKeys.end() == std::find(supported_configKeys.begin(), supported_configKeys.end(), name))
            IE_THROW() << "Unsupported config key: " << name;
        if (name == CONFIG_KEY(AUTO_BATCH_DEVICE_CONFIG)) {
            ParseBatchDevice(val);
        } else if (name == CONFIG_KEY(AUTO_BATCH_TIMEOUT)) {
            try {
                auto t = std::stoi(val);
                if (t < 0)
                    IE_THROW(ParameterMismatch);
            } catch (const std::exception&) {
                IE_THROW(ParameterMismatch)
                    << " Expecting unsigned int value for " << CONFIG_KEY(AUTO_BATCH_TIMEOUT) << " got " << val;
            }
        }
    }
}

void AutoBatchInferencePlugin::SetConfig(const std::map<std::string, std::string>& config) {
    CheckConfig(config);
    for (auto&& kvp : config) {
        _config[kvp.first] = kvp.second;
    }
}

static const Version version = {{2, 1}, CI_BUILD_NUMBER, "AutoBatchPlugin"};
IE_DEFINE_PLUGIN_CREATE_FUNCTION(AutoBatchInferencePlugin, version)

AutoBatchInferencePlugin::AutoBatchInferencePlugin() {
    _pluginName = "BATCH";
    _config[CONFIG_KEY(AUTO_BATCH_TIMEOUT)] = "1000";  // default value, in ms
}

InferenceEngine::Parameter AutoBatchInferencePlugin::GetMetric(
    const std::string& name,
    const std::map<std::string, InferenceEngine::Parameter>& options) const {
    if (name == METRIC_KEY(SUPPORTED_METRICS)) {
        std::vector<std::string> metrics;
        metrics.push_back(METRIC_KEY(SUPPORTED_METRICS));
        metrics.push_back(METRIC_KEY(FULL_DEVICE_NAME));
        metrics.push_back(METRIC_KEY(SUPPORTED_CONFIG_KEYS));
        IE_SET_METRIC_RETURN(SUPPORTED_METRICS, metrics);
    } else if (name == METRIC_KEY(FULL_DEVICE_NAME)) {
        IE_SET_METRIC_RETURN(FULL_DEVICE_NAME, _pluginName);
    } else if (name == METRIC_KEY(SUPPORTED_CONFIG_KEYS)) {
        IE_SET_METRIC_RETURN(SUPPORTED_CONFIG_KEYS, supported_configKeys);
    } else {
        IE_THROW(NotFound) << "Unsupported metric key " << name;
    }
}

IExecutableNetworkInternal::Ptr AutoBatchInferencePlugin::LoadExeNetworkImpl(
    const InferenceEngine::CNNNetwork& network,
    const std::map<std::string, std::string>& config) {
    return LoadNetworkImpl(network, nullptr, config);
}

InferenceEngine::IExecutableNetworkInternal::Ptr AutoBatchInferencePlugin::LoadNetworkImpl(
    const InferenceEngine::CNNNetwork& network,
    const std::shared_ptr<InferenceEngine::RemoteContext> ctx,
    const std::map<std::string, std::string>& config) {
    auto core = GetCore();
    if (core == nullptr) {
        IE_THROW() << "Please, work with Auto-Batching device via InferencEngine::Core object";
    }
    auto fullConfig = mergeConfigs(_config, config);
    auto device_batch = fullConfig.find(CONFIG_KEY(AUTO_BATCH_DEVICE_CONFIG));
    if (device_batch == fullConfig.end()) {
        IE_THROW() << "KEY_AUTO_BATCH key is not set for BATCH device";
    }
    auto metaDevice = ParseMetaDevice(device_batch->second, fullConfig);
    const auto& deviceName = metaDevice.deviceName;
    const auto& deviceConfig = metaDevice.config;
    auto deviceConfigNoAutoBatch = deviceConfig;
    // avoid recursive auto-batching
    deviceConfigNoAutoBatch[CONFIG_KEY(ALLOW_AUTO_BATCHING)] = CONFIG_VALUE(NO);

    std::set<std::string> batched_inputs;
    std::set<std::string> batched_outputs;
    // check that the auto-batching is applicable in general
    try {
        // if applicable, the Auto-Batching is implicitly enabled via the performance hints
        const auto tput = CONFIG_VALUE(THROUGHPUT);
        const bool bTputInPlg = core->GetConfig(deviceName, CONFIG_KEY(PERFORMANCE_HINT)).as<std::string>() == tput;
        const auto& mode = deviceConfig.find(CONFIG_KEY(PERFORMANCE_HINT));
        const bool bTputInLoadCfg = (mode != deviceConfig.end() && mode->second == tput);
        // if the auto-batching is enabled implicitly, check the dims carefully, to avoid outstanding failures
        const bool check_dims = (bTputInPlg || bTputInLoadCfg);
        CNNNetwork clonedNetwork(InferenceEngine::details::cloneNetwork(network));
        auto function = clonedNetwork.getFunction();
        // find the batch dim
        ov::pass::Manager m;
        m.register_pass<ov::pass::InitNodeInfo>();
        m.register_pass<ov::pass::FindBatch>(false, check_dims);
        m.run_passes(function);
        // do not reshape/re-batch originally batched networks and when there are no inputs with the N* layouts
        // input(s) should have the batch dim as the first dim (current limitation of the auto-batching impl)
        const auto& params = function->get_parameters();
        for (size_t input_id = 0; input_id < params.size(); input_id++) {
            const auto& input = params[input_id];
            const auto& shape = input->get_partial_shape();
            // currently no plugin support batched execution for dynamic networks
            if (shape.is_dynamic())
                IE_THROW(NotImplemented) << "Auto-batching does not support dynamic networks!";
            // check the batch dim: either 0th (and the original batch size of 1) or none
            if (shape.size() && ov::DimensionTracker::get_label(shape[0])) {
                const auto& static_shape = input->get_shape();
                if (static_shape[0] != 1)
                    IE_THROW(NotImplemented) << "Auto-batching does not reshape/re-batch originally batched networks!";
                batched_inputs.insert(
                    ov::op::util::get_ie_output_name(params[input_id]->output(0)));  // batched dim for the input
            } else {
                // if the 0-th dim is not for the batch, then we support only the case when NONE dimension is batch
                for (size_t s = 1; s < shape.size(); s++)
                    if (ov::DimensionTracker::get_label(shape[s]))
                        IE_THROW(NotImplemented)
                            << "Auto-batching operates only networks with inputs/outputs batched by 0th dimension";
            }
        }
        const auto& results = function->get_results();
        for (size_t output_id = 0; output_id < results.size(); output_id++) {
            const auto& output = results[output_id];
            const auto& shape = output->get_output_partial_shape(0);
            if (shape.is_dynamic())
                IE_THROW(NotImplemented) << "Auto-batching does not support dynamic networks!";
            // check the batch dim: either 0th (and the original batch size of 1) or none
            if (shape.size() && ov::DimensionTracker::get_label(shape[0])) {
                if (shape[0] != 1)
                    IE_THROW(NotImplemented) << "Auto-batching does not reshape/re-batch originally batched networks!";
                const auto& node = output->input_value(0);
                batched_outputs.insert(
                    ov::op::util::get_ie_output_name(ov::Output<const ov::Node>(node.get_node(), node.get_index())));
            } else {
                // if the 0-th dim is not for the batch, then we support only the case when NONE dimension is batch
                for (size_t s = 1; s < shape.size(); s++)
                    if (ov::DimensionTracker::get_label(shape[s]))
                        IE_THROW(NotImplemented)
                            << "Auto-batching operates only networks with outputs batched by 0th dimension";
            }
        }
        if (!batched_inputs.size() || !batched_outputs.size())
            IE_THROW(NotImplemented)
                << "Auto-batching supports only networks with inputs/outputs featuring batched dim!";
    } catch (const InferenceEngine::Exception&) {
        metaDevice.batchForDevice = 1;
    }

    if (!metaDevice.batchForDevice) {
        unsigned int requests = 0;
        // batch size is not set explicitly via device name e.g. BATCH:GPU(4)
        // let's query the optimal batch size
        std::map<std::string, InferenceEngine::Parameter> options;
        options["MODEL_PTR"] = std::const_pointer_cast<ngraph::Function>(network.getFunction());
        auto optBatchSize = core->GetMetric(deviceName, METRIC_KEY(OPTIMAL_BATCH_SIZE), options).as<unsigned int>();
        auto res = core->GetConfig(deviceName, CONFIG_KEY(PERFORMANCE_HINT_NUM_REQUESTS)).as<std::string>();
        requests = PerfHintsConfig::CheckPerformanceHintRequestValue(res);
        const auto& reqs = config.find(CONFIG_KEY(PERFORMANCE_HINT_NUM_REQUESTS));
        if (reqs != config.end())
            requests = static_cast<unsigned int>(PerfHintsConfig::CheckPerformanceHintRequestValue(reqs->second));
        if (requests)
            optBatchSize = std::max(1u, std::min(requests, optBatchSize));
        if (optBatchSize > 2)  // batching is usually in-efficient for batch<4 (as batch1 kernels are heavily optimized)
            metaDevice.batchForDevice = optBatchSize;
        else
            metaDevice.batchForDevice = 1;
    }

    auto report_footprint = [](std::shared_ptr<ICore> pCore, std::string device) -> size_t {
        size_t footprint = 0;
        // TODO: use the per-network metric (22.2) rather than plugin-level
        auto stats =
            pCore->GetMetric(device, ov::intel_gpu::memory_statistics.name()).as<std::map<std::string, uint64_t>>();
        for (auto s : stats)
            footprint += s.second;
        return footprint;
    };

    size_t batch1_footprint = 0;
    if (deviceName.find("GPU") != std::string::npos)
        batch1_footprint = report_footprint(core, deviceName);
    auto executableNetworkWithoutBatch = ctx ? core->LoadNetwork(network, ctx, deviceConfigNoAutoBatch)
                                             : core->LoadNetwork(network, deviceName, deviceConfigNoAutoBatch);
    if (deviceName.find("GPU") != std::string::npos) {
        batch1_footprint = report_footprint(core, deviceName) - batch1_footprint;
        if (batch1_footprint) {
            const auto total_mem =
                GetCore()->GetMetric(deviceName, GPU_METRIC_KEY(DEVICE_TOTAL_MEM_SIZE)).as<uint64_t>();
            const int estimated_batch = static_cast<int>((total_mem - batch1_footprint) / batch1_footprint);
            int closest = static_cast<int>(pow(2, floor(log(estimated_batch) / log(2))));
            closest = std::max(1, closest);
            metaDevice.batchForDevice = std::min(metaDevice.batchForDevice, closest);
        }
    }
    // auto-batch settings
    std::unordered_map<std::string, InferenceEngine::Parameter> networkConfig;
    for (auto c : fullConfig) {
        if (supported_configKeys.end() != std::find(supported_configKeys.begin(), supported_configKeys.end(), c.first))
            networkConfig.insert(c);
    }

    InferenceEngine::SoExecutableNetworkInternal executableNetworkWithBatch;
    if (metaDevice.batchForDevice > 1 && batched_inputs.size()) {
        try {
            CNNNetwork reshaped(InferenceEngine::details::cloneNetwork(network));
            ICNNNetwork::InputShapes shapes = reshaped.getInputShapes();
            for (const auto& input : batched_inputs)
                shapes[input][0] = metaDevice.batchForDevice;
            reshaped.reshape(shapes);
            executableNetworkWithBatch = ctx ? core->LoadNetwork(reshaped, ctx, deviceConfigNoAutoBatch)
                                             : core->LoadNetwork(reshaped, deviceName, deviceConfigNoAutoBatch);
        } catch (const InferenceEngine::Exception&) {
            metaDevice.batchForDevice = 1;
        }
    }

    return std::make_shared<AutoBatchExecutableNetwork>(executableNetworkWithBatch,
                                                        executableNetworkWithoutBatch,
                                                        metaDevice,
                                                        networkConfig,
                                                        batched_inputs,
                                                        batched_outputs);
}

InferenceEngine::IExecutableNetworkInternal::Ptr AutoBatchInferencePlugin::LoadExeNetworkImpl(
    const InferenceEngine::CNNNetwork& network,
    const std::shared_ptr<InferenceEngine::RemoteContext>& context,
    const std::map<std::string, std::string>& config) {
    return LoadNetworkImpl(network, context, config);
}

InferenceEngine::QueryNetworkResult AutoBatchInferencePlugin::QueryNetwork(
    const InferenceEngine::CNNNetwork& network,
    const std::map<std::string, std::string>& config) const {
    auto core = GetCore();
    if (!core)
        return InferenceEngine::QueryNetworkResult();
    auto cfg = config;
    for (auto c : cfg) {
        if (c.first == CONFIG_KEY(AUTO_BATCH_DEVICE_CONFIG)) {
            auto val = c.second;
            cfg.erase(c.first);
            auto metaDevice = ParseMetaDevice(val, cfg);
            return core->QueryNetwork(network, metaDevice.deviceName, cfg);
        }
    }
    IE_THROW() << "Value for KEY_AUTO_BATCH is not set";
}
}  // namespace AutoBatchPlugin
