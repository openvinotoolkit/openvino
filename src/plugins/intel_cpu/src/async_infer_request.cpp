// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "async_infer_request.h"

#include <memory>
#include <vector>

#include "infer_request.h"
#include "openvino/runtime/iasync_infer_request.hpp"
#include "openvino/runtime/iinfer_request.hpp"
#include "openvino/runtime/threading/istreams_executor.hpp"
#include "openvino/runtime/threading/itask_executor.hpp"

namespace {

// Unlike the base IAsyncInferRequest's default ImmediateStreamsExecutor (which dispatches to a worker thread via
// run_and_wait() whenever the underlying streams executor has more than one stream), CallerThreadStageExecutor
// always runs the stage task directly on the calling thread via execute(). This is the dispatch behavior the
// CPU_MULTI_APP_THREAD_SYNC_EXECUTION / optimized-single-stream features require: the application thread that
// called infer() must do the inference work itself instead of handing it off to a streams-executor worker.
//
// Because this executor is only ever installed as the first (and only) stage of m_sync_pipeline, the stage task
// handed to run() is always the lambda produced by IAsyncInferRequest::make_next_stage_task(), which already
// wraps the real work in its own try/catch before any executor ever sees it. So unlike a raw, unwrapped lambda
// passed straight to CPUStreamsExecutor::execute()/Defer(), nothing can throw out of this task and be swallowed
// by the streams executor's own exception handling.
//
// Known limitation (deferred, see PR review item F-5): this does NOT save/restore the calling thread's CPU
// affinity mask, OpenMP max-thread count, or MXCSR FTZ/DAZ bits around execute(). If the streams executor's
// dispatch mutates any of those as a side effect on this thread, that mutation currently leaks back into the
// application thread once infer() returns. A RAII CallerThreadScope guard around this call is the proposed
// future fix; not implemented here.
class CallerThreadStageExecutor : public ov::threading::ITaskExecutor {
public:
    explicit CallerThreadStageExecutor(std::shared_ptr<ov::threading::IStreamsExecutor> stream_executor)
        : m_stream_executor(std::move(stream_executor)) {}

    void run(ov::threading::Task task) override {
        m_stream_executor->execute(std::move(task));
    }

private:
    std::shared_ptr<ov::threading::IStreamsExecutor> m_stream_executor;
};

}  // namespace

ov::intel_cpu::AsyncInferRequest::AsyncInferRequest(
    const std::shared_ptr<IInferRequest>& request,
    const std::shared_ptr<ov::threading::ITaskExecutor>& task_executor,
    const std::shared_ptr<ov::threading::ITaskExecutor>& callback_executor,
    const bool is_optimized_single_stream,
    const bool run_sync_in_caller_thread)
    : ov::IAsyncInferRequest(request, task_executor, callback_executor),
      m_internal_request(request) {
    static_cast<SyncInferRequest*>(request.get())->set_async_request(this);
    if (is_optimized_single_stream || run_sync_in_caller_thread) {
        if (auto stream_executor = std::dynamic_pointer_cast<ov::threading::IStreamsExecutor>(task_executor)) {
            m_sync_pipeline = {{std::make_shared<CallerThreadStageExecutor>(std::move(stream_executor)),
                                [this]() {
                                    m_internal_request->infer();
                                }}};
        }
    }
}

ov::intel_cpu::AsyncInferRequest::~AsyncInferRequest() {
    if (m_has_sub_infers) {
        m_sub_infer_requests.clear();
    }
    stop_and_wait();
}

void ov::intel_cpu::AsyncInferRequest::throw_if_canceled() const {
    check_cancelled_state();
}

void ov::intel_cpu::AsyncInferRequest::setSubInferRequest(
    const std::vector<std::shared_ptr<IAsyncInferRequest>>& requests) {
    m_sub_infer_requests = requests;
}
