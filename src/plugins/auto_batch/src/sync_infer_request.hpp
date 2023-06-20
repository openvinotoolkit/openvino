// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///////////////////////////////////////////////////////////////////////////////////////////////////
#pragma once

#include "compiled_model.hpp"
#include "cpp_interfaces/interface/ie_iinfer_request_internal.hpp"

namespace ov {
namespace autobatch_plugin {

class AutoBatchInferRequest : public InferenceEngine::IInferRequestInternal {
public:
    using Ptr = std::shared_ptr<AutoBatchInferRequest>;
    explicit AutoBatchInferRequest(const InferenceEngine::InputsDataMap& networkInputs,
                                   const InferenceEngine::OutputsDataMap& networkOutputs,
                                   CompiledModel::WorkerInferRequest& workerRequestPtr,
                                   int batch_id,
                                   int num_batch,
                                   const std::set<std::string>& batchedIntputs,
                                   const std::set<std::string>& batchedOutputs);
    explicit AutoBatchInferRequest(const std::vector<std::shared_ptr<const ov::Node>>& inputs,
                                   const std::vector<std::shared_ptr<const ov::Node>>& outputs,
                                   CompiledModel::WorkerInferRequest& workerRequestPtr,
                                   int batch_id,
                                   int num_batch,
                                   const std::set<std::string>& batchedIntputs,
                                   const std::set<std::string>& batchedOutputs);

    // Batch-Device impl specific: sets the data (blobs from the device request to the batched device request)
    void SetBlobsToAnotherRequest(InferenceEngine::SoIInferRequestInternal& req);
    void CopyInputsIfNeeded();
    void CopyOutputsIfNeeded();
    CompiledModel::WorkerInferRequest& _myBatchedRequestWrapper;
    std::exception_ptr _exceptionPtr;
    enum eExecutionFlavor : uint8_t {
        NOT_EXECUTED,
        BATCH_EXECUTED,
        TIMEOUT_EXECUTED
    } _wasBatchedRequestUsed = eExecutionFlavor::NOT_EXECUTED;

protected:
    void CopyBlobIfNeeded(InferenceEngine::Blob::CPtr src, InferenceEngine::Blob::Ptr dst, bool bInput);
    void ShareBlobsWithBatchRequest(const std::set<std::string>& batchedIntputs,
                                    const std::set<std::string>& batchedOutputs);
    size_t _batchId;
    size_t _batchSize;
};
}  // namespace autobatch_plugin
}  // namespace ov