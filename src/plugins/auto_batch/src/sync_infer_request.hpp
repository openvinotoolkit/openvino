// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///////////////////////////////////////////////////////////////////////////////////////////////////
#pragma once

#include "compiled_model.hpp"
#include "openvino/runtime/isync_infer_request.hpp"

namespace ov {
namespace autobatch_plugin {

class SyncInferRequest : public ov::ISyncInferRequest {
public:
    SyncInferRequest(const std::shared_ptr<const ov::autobatch_plugin::CompiledModel>& compiled_model,
                     const std::shared_ptr<ov::autobatch_plugin::CompiledModel::WorkerInferRequest>& worker_request,
                     int batch_id,
                     int num_batch,
                     const std::set<std::size_t>& batched_inputs = {},
                     const std::set<std::size_t>& batched_outputs = {});

    // Batch-Device impl specific: sets the data (blobs from the device request to the batched device request)
    void set_tensors_to_another_request(ov::SoPtr<ov::IAsyncInferRequest>& req);

    void copy_inputs_if_needed();

    void copy_outputs_if_needed();

    void infer() override;

    std::vector<ov::SoPtr<ov::IVariableState>> query_state() const override;

    std::vector<ov::ProfilingInfo> get_profiling_info() const override;

    std::shared_ptr<ov::autobatch_plugin::CompiledModel::WorkerInferRequest> m_batched_request_wrapper;

    std::exception_ptr m_exception_ptr;

    enum eExecutionFlavor : uint8_t {
        NOT_EXECUTED,
        BATCH_EXECUTED,
        TIMEOUT_EXECUTED
    } m_batched_request_status = eExecutionFlavor::NOT_EXECUTED;

    size_t get_batch_size() const;

protected:
    void copy_tensor_if_needed(const ov::SoPtr<ov::ITensor>& src, ov::SoPtr<ov::ITensor>& dst, const bool bInput);

    void share_tensors_with_batched_req(const std::set<std::size_t>& batched_inputs,
                                        const std::set<std::size_t>& batched_outputs);

    size_t m_batch_id;

    size_t m_batch_size;
};
}  // namespace autobatch_plugin
}  // namespace ov