// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///////////////////////////////////////////////////////////////////////////////////////////////////
#pragma once

#include "openvino/runtime/iasync_infer_request.hpp"
#include "sync_infer_request.hpp"

namespace ov {
namespace autobatch_plugin {
class AsyncInferRequest : public ov::IAsyncInferRequest {
public:
    AsyncInferRequest(const std::shared_ptr<SyncInferRequest>& request,
                      const ov::SoPtr<ov::IAsyncInferRequest>& request_without_batch,
                      const std::shared_ptr<ov::threading::ITaskExecutor>& callback_executor);

    void infer_thread_unsafe() override;

    virtual ~AsyncInferRequest();

    std::vector<ov::ProfilingInfo> get_profiling_info() const override;

    std::vector<ov::SoPtr<ov::IVariableState>> query_state() const override;

    std::shared_ptr<ov::autobatch_plugin::SyncInferRequest> m_sync_request;

    void set_tensor(const ov::Output<const ov::Node>& port, const ov::SoPtr<ov::ITensor>& tensor) override;

    void set_tensors(const ov::Output<const ov::Node>& port,
                     const std::vector<ov::SoPtr<ov::ITensor>>& tensors) override;

    ov::SoPtr<ov::IAsyncInferRequest> m_request_without_batch;
};
}  // namespace autobatch_plugin
}  // namespace ov