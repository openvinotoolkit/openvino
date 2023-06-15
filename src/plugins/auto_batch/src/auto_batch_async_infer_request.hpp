// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "auto_batch_sync_infer_request.hpp"
#include "openvino/runtime/iasync_infer_request.hpp"

namespace ov {
namespace autobatch_plugin {
class AsyncInferRequest : public ov::IAsyncInferRequest {
public:
    AsyncInferRequest(const std::shared_ptr<ov::autobatch_plugin::SyncInferRequest>& request,
                      std::shared_ptr<ov::IAsyncInferRequest> inferRequestWithoutBatch,
                      const std::shared_ptr<ov::threading::ITaskExecutor>& callback_executor);
    virtual ~AsyncInferRequest();
    void infer_thread_unsafe() override;
    std::vector<ov::ProfilingInfo> get_profiling_info() const override;
    std::shared_ptr<ov::autobatch_plugin::SyncInferRequest> m_sync_request;
    std::shared_ptr<ov::IAsyncInferRequest> m_inferRequestWithoutBatch;
};
}  // namespace autobatch_plugin
}  // namespace ov