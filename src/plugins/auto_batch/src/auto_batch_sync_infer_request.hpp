// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "auto_batch_compiled_model.hpp"
#include "openvino/runtime/isync_infer_request.hpp"

namespace ov {
namespace autobatch_plugin {
// forward declaration
class CompiledModel;
class SyncInferRequest : public ov::ISyncInferRequest {
public:
    explicit SyncInferRequest(const std::shared_ptr<const ov::autobatch_plugin::CompiledModel>& compiled_model,
                              std::shared_ptr<ov::autobatch_plugin::CompiledModel::WorkerInferRequest> workerRequest,
                              int batch_id,
                              int num_batch,
                              const std::set<std::string>& batchedInputs,
                              const std::set<std::string>& batchedOutputs);

    void infer() override;
    std::vector<std::shared_ptr<ov::IVariableState>> query_state() const override;
    std::vector<ov::ProfilingInfo> get_profiling_info() const override;
    void SetBlobsToAnotherRequest(std::shared_ptr<ov::IAsyncInferRequest>& req);
    void CopyInputsIfNeeded();
    void CopyOutputsIfNeeded();
    std::shared_ptr<ov::autobatch_plugin::CompiledModel::WorkerInferRequest> _myBatchedRequestWrapper;
    std::exception_ptr _exceptionPtr;
    enum eExecutionFlavor : uint8_t {
        NOT_EXECUTED,
        BATCH_EXECUTED,
        TIMEOUT_EXECUTED
    } _wasBatchedRequestUsed = eExecutionFlavor::NOT_EXECUTED;

protected:
    void CopyBlobIfNeeded(const ov::Tensor& src, ov::Tensor& dst, bool bInput);
    void ShareBlobsWithBatchRequest(const std::set<std::string>& batchedInputs,
                                    const std::set<std::string>& batchedOutputs);
    size_t _batchId;
    size_t _batchSize;
};
}  // namespace autobatch_plugin
}  // namespace ov