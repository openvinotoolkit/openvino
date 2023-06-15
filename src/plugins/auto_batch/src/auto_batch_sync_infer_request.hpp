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

    void set_tensors_to_another_request(std::shared_ptr<ov::IAsyncInferRequest>& req);

    void copy_inputs_if_needed();

    void copy_outputs_if_needed();

    enum eExecutionFlavor : uint8_t {
        NOT_EXECUTED,
        BATCH_EXECUTED,
        TIMEOUT_EXECUTED
    } m_batched_req_used = eExecutionFlavor::NOT_EXECUTED;

    std::shared_ptr<ov::autobatch_plugin::CompiledModel::WorkerInferRequest> m_batched_request_wrapper;

    std::exception_ptr m_exception_ptr;

protected:
    void copy_tensor_if_needed(const ov::Tensor& src, ov::Tensor& dst, bool bInput);
    void share_tensors_with_batched_req(const std::set<std::string>& batchedInputs,
                                        const std::set<std::string>& batchedOutputs);
    size_t m_batch_id;
    size_t m_batch_size;
};
}  // namespace autobatch_plugin
}  // namespace ov