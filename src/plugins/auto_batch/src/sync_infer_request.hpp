// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///////////////////////////////////////////////////////////////////////////////////////////////////
#pragma once

#include "compiled_model.hpp"
#include "cpp_interfaces/interface/ie_iinfer_request_internal.hpp"

namespace ov {
namespace autobatch_plugin {

class SyncInferRequest : public InferenceEngine::IInferRequestInternal {
public:
    using Ptr = std::shared_ptr<SyncInferRequest>;
    explicit SyncInferRequest(const InferenceEngine::InputsDataMap& networkInputs,
                                   const InferenceEngine::OutputsDataMap& networkOutputs,
                                   CompiledModel::WorkerInferRequest& workerRequestPtr,
                                   int batch_id,
                                   int num_batch,
                                   const std::set<std::string>& batchedIntputs,
                                   const std::set<std::string>& batchedOutputs);

    explicit SyncInferRequest(const std::vector<std::shared_ptr<const ov::Node>>& inputs,
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

    CompiledModel::WorkerInferRequest& m_batched_request_wrapper;

    std::exception_ptr m_exceptionPtr;

    enum eExecutionFlavor : uint8_t {
        NOT_EXECUTED,
        BATCH_EXECUTED,
        TIMEOUT_EXECUTED
    } m_batched_request_status = eExecutionFlavor::NOT_EXECUTED;

protected:
    void CopyBlobIfNeeded(InferenceEngine::Blob::CPtr src, InferenceEngine::Blob::Ptr dst, bool bInput);

    void ShareBlobsWithBatchRequest(const std::set<std::string>& batchedIntputs,
                                    const std::set<std::string>& batchedOutputs);
    size_t m_batch_id;

    size_t m_batch_size;
};
}  // namespace autobatch_plugin
}  // namespace ov