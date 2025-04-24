// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///////////////////////////////////////////////////////////////////////////////////////////////////
#pragma once

#include "schedule.hpp"
#include "infer_request.hpp"

namespace ov {
namespace auto_plugin {
// ! [async_infer_request:header]
class AsyncInferRequest : public ov::IAsyncInferRequest {
public:
    AsyncInferRequest(const Schedule::Ptr& schedule,
                      const std::shared_ptr<ov::auto_plugin::InferRequest>& request,
                      const std::shared_ptr<ov::threading::ITaskExecutor>& callback_executor);

    ~AsyncInferRequest();
    void infer_thread_unsafe() override;
    std::vector<ov::ProfilingInfo> get_profiling_info() const override;
private:
    Schedule::Ptr       m_schedule;
    WorkerInferRequest* m_worker_inferrequest = nullptr;
    ISyncInferPtr       m_inferrequest;
};

}  // namespace auto_plugin
}  // namespace ov
