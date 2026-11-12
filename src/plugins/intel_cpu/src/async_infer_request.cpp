// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "async_infer_request.h"

#include <memory>
#include <vector>

#include "config.h"
#include "infer_request.h"
#include "openvino/core/parallel.hpp"
#include "openvino/core/visibility.hpp"
#include "openvino/runtime/iasync_infer_request.hpp"
#include "openvino/runtime/iinfer_request.hpp"
#include "openvino/runtime/threading/istreams_executor.hpp"
#include "openvino/runtime/threading/itask_executor.hpp"
#include "utils/denormals.hpp"

// _mm_getcsr()/_mm_setcsr() (and <immintrin.h>) are only actually available where SSE is present.
// OPENVINO_ARCH_X86 is intentionally also defined for 32-bit/ILP32 targets like __wasm32__ (see
// visibility.hpp), which do NOT have these intrinsics, so that macro alone is not a safe guard here.
// Mirror the precise condition denormals.hpp itself uses to decide whether MXCSR intrinsics are usable.
#if defined(OPENVINO_ARCH_X86_64) || defined(__SSE__) || (defined(_M_IX86_FP) && _M_IX86_FP >= 1)
#    define OV_CPU_HAS_MXCSR_INTRINSICS 1
#    include <immintrin.h>
#else
#    define OV_CPU_HAS_MXCSR_INTRINSICS 0
#endif

namespace {

// RAII guard that snapshots, on entry, the pieces of thread-local CPU/runtime state that
// CPUStreamsExecutor::execute()/Defer() may mutate as a side effect of lazily constructing its per-thread Stream
// (see PR review item F-5), and restores them on the calling application thread when the guard goes out of scope:
//   - MXCSR (FTZ/DAZ denormal-handling bits): the raw register is snapshotted before applying `denormals_opt_mode`
//     (matching what Plugin::compile_model() does on the compiling thread, see plugin.cpp) and restored to
//     whatever the calling application thread had *before* this inline inference call, once this guard goes out
//     of scope. This both (a) makes CPU_DENORMALS_OPTIMIZATION actually take effect for inference that runs
//     inline on an application thread -- previously only the thread that called compile_model() ever got
//     FTZ/DAZ set, so any other app thread calling infer() directly via multi-app-thread-sync never benefited
//     from it -- and (b) still protects the calling thread from being left with that state after infer() returns.
//   - OpenMP max-thread count: on OV_THREAD_OMP builds, Stream's constructor unconditionally calls
//     omp_set_num_threads() on whichever thread constructs it. For CPU_MULTI_APP_THREAD_SYNC_EXECUTION /
//     optimized-single-stream, that thread is this application thread the first time it calls infer(), so without
//     this guard the app thread's own OpenMP thread count would be silently changed for good.
// Note: CPU affinity-mask pinning on this thread is NOT handled here and remains a known, separately-tracked
// limitation (out of scope for this fix).
class CallerThreadStateGuard {
public:
    explicit CallerThreadStateGuard(ov::intel_cpu::Config::DenormalsOptMode denormals_opt_mode) {
#if OV_CPU_HAS_MXCSR_INTRINSICS
        m_mxcsr = _mm_getcsr();
        if (denormals_opt_mode == ov::intel_cpu::Config::DenormalsOptMode::DO_On) {
            ov::intel_cpu::flush_to_zero(true);
            ov::intel_cpu::denormals_as_zero(true);
        } else if (denormals_opt_mode == ov::intel_cpu::Config::DenormalsOptMode::DO_Off) {
            ov::intel_cpu::flush_to_zero(false);
            ov::intel_cpu::denormals_as_zero(false);
        }
        // DO_Keep: leave this thread's current FTZ/DAZ bits untouched, matching compile_model()'s own behavior.
#endif
#if OV_THREAD == OV_THREAD_OMP
        m_omp_max_threads = omp_get_max_threads();
#endif
    }

    CallerThreadStateGuard(const CallerThreadStateGuard&) = delete;
    CallerThreadStateGuard& operator=(const CallerThreadStateGuard&) = delete;

    ~CallerThreadStateGuard() {
#if OV_CPU_HAS_MXCSR_INTRINSICS
        _mm_setcsr(m_mxcsr);
#endif
#if OV_THREAD == OV_THREAD_OMP
        omp_set_num_threads(m_omp_max_threads);
#endif
    }

private:
#if OV_CPU_HAS_MXCSR_INTRINSICS
    unsigned int m_mxcsr = 0;
#endif
#if OV_THREAD == OV_THREAD_OMP
    int m_omp_max_threads = 1;
#endif
};

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
class CallerThreadStageExecutor : public ov::threading::ITaskExecutor {
public:
    CallerThreadStageExecutor(std::shared_ptr<ov::threading::IStreamsExecutor> stream_executor,
                              ov::intel_cpu::Config::DenormalsOptMode denormals_opt_mode)
        : m_stream_executor(std::move(stream_executor)),
          m_denormals_opt_mode(denormals_opt_mode) {}

    void run(ov::threading::Task task) override {
        // Apply CPU_DENORMALS_OPTIMIZATION (if requested) and protect the calling application thread's own
        // OpenMP/MXCSR state from being leaked into by inline inference execution. See CallerThreadStateGuard
        // above for details.
        CallerThreadStateGuard state_guard(m_denormals_opt_mode);
        m_stream_executor->execute(std::move(task));
    }

private:
    std::shared_ptr<ov::threading::IStreamsExecutor> m_stream_executor;
    ov::intel_cpu::Config::DenormalsOptMode m_denormals_opt_mode;
};

}  // namespace

ov::intel_cpu::AsyncInferRequest::AsyncInferRequest(
    const std::shared_ptr<IInferRequest>& request,
    const std::shared_ptr<ov::threading::ITaskExecutor>& task_executor,
    const std::shared_ptr<ov::threading::ITaskExecutor>& callback_executor,
    const bool is_optimized_single_stream,
    const bool run_sync_in_caller_thread,
    const Config::DenormalsOptMode denormals_opt_mode)
    : ov::IAsyncInferRequest(request, task_executor, callback_executor),
      m_internal_request(request) {
    static_cast<SyncInferRequest*>(request.get())->set_async_request(this);
    if (is_optimized_single_stream || run_sync_in_caller_thread) {
        if (auto stream_executor = std::dynamic_pointer_cast<ov::threading::IStreamsExecutor>(task_executor)) {
            m_sync_pipeline = {
                {std::make_shared<CallerThreadStageExecutor>(std::move(stream_executor), denormals_opt_mode), [this]() {
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
