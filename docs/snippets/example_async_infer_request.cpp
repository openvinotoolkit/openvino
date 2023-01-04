// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <threading/ie_itask_executor.hpp>
#include <cpp_interfaces/impl/ie_infer_async_request_thread_safe_default.hpp>
#include <memory>

using namespace InferenceEngine;

class AcceleratorSyncRequest : public IInferRequestInternal {
public:
    using Ptr = std::shared_ptr<AcceleratorSyncRequest>;

    void preprocess();
    void write_to_device();
    void run_on_device();
    void read_from_device();
    void post_process();
};

// ! [async_infer_request:define_pipeline]
// Inherits from AsyncInferRequestThreadSafeDefault
class AcceleratorAsyncInferRequest : public AsyncInferRequestThreadSafeDefault {
    // Store the pointer to the synchronous request and five executors
    AcceleratorAsyncInferRequest(const AcceleratorSyncRequest::Ptr& syncRequest,
                                 const ITaskExecutor::Ptr& preprocessExecutor,
                                 const ITaskExecutor::Ptr& writeToDeviceExecutor,
                                 const ITaskExecutor::Ptr& runOnDeviceExecutor,
                                 const ITaskExecutor::Ptr& readFromDeviceExecutor,
                                 const ITaskExecutor::Ptr& postProcessExecutor) :
    AsyncInferRequestThreadSafeDefault(syncRequest, nullptr, nullptr),
    _accSyncRequest{syncRequest},
    _preprocessExecutor{preprocessExecutor},
    _writeToDeviceExecutor{writeToDeviceExecutor},
    _runOnDeviceExecutor{runOnDeviceExecutor},
    _readFromDeviceExecutor{readFromDeviceExecutor},
    _postProcessExecutor{postProcessExecutor}
    {
        // Five pipeline stages of synchronous infer request are run by different executors
        _pipeline = {
            { _preprocessExecutor , [this] {
                _accSyncRequest->preprocess();
            }},
            { _writeToDeviceExecutor , [this] {
                _accSyncRequest->write_to_device();
            }},
            { _runOnDeviceExecutor , [this] {
                _accSyncRequest->run_on_device();
            }},
            { _readFromDeviceExecutor , [this] {
                _accSyncRequest->read_from_device();
            }},
            { _postProcessExecutor , [this] {
                _accSyncRequest->post_process();
            }},
        };
    }

    // As all stages use _accSyncRequest member we should wait for all stages tasks before the destructor destroy this member.
    ~AcceleratorAsyncInferRequest() {
        StopAndWait();
    }

    AcceleratorSyncRequest::Ptr _accSyncRequest;
    ITaskExecutor::Ptr _preprocessExecutor, _writeToDeviceExecutor, _runOnDeviceExecutor, _readFromDeviceExecutor, _postProcessExecutor;
};
// ! [async_infer_request:define_pipeline]
