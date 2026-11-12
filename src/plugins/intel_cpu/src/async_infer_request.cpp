// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "async_infer_request.h"

#include <memory>
#include <utility>
#include <vector>

#include "config.h"
#include "infer_request.h"
#include "openvino/core/parallel.hpp"
#include "openvino/core/visibility.hpp"
#include "openvino/runtime/iasync_infer_request.hpp"
#include "openvino/runtime/iinfer_request.hpp"
#include "openvino/runtime/threading/istreams_executor.hpp"
#include "openvino/runtime/threading/itask_executor.hpp"

// MXCSR intrinsics need SSE, which OPENVINO_ARCH_X86 alone doesn't guarantee (e.g. wasm32); mirror
// denormals.hpp's own guard condition.
#if defined(OPENVINO_ARCH_X86_64) || defined(__SSE__) || (defined(_M_IX86_FP) && _M_IX86_FP >= 1)
#    define OV_CPU_HAS_MXCSR_INTRINSICS 1
#    include <immintrin.h>

#    include "utils/denormals.hpp"
#else
#    define OV_CPU_HAS_MXCSR_INTRINSICS 0
#endif

namespace {

// RAII guard: snapshots this thread's MXCSR (FTZ/DAZ) and OpenMP max-thread count on entry, applies
// denormals_opt_mode, and restores both on scope exit. Protects a caller application thread from
// having its own state mutated by lazy Stream construction when inference runs inline on it.
class CallerThreadStateGuard {
public:
    explicit CallerThreadStateGuard(ov::intel_cpu::Config::DenormalsOptMode denormals_opt_mode) {
#if OV_CPU_HAS_MXCSR_INTRINSICS
        if (denormals_opt_mode == ov::intel_cpu::Config::DenormalsOptMode::DO_On) {
            ov::intel_cpu::flush_to_zero(true);
            ov::intel_cpu::denormals_as_zero(true);
        } else if (denormals_opt_mode == ov::intel_cpu::Config::DenormalsOptMode::DO_Off) {
            ov::intel_cpu::flush_to_zero(false);
            ov::intel_cpu::denormals_as_zero(false);
        }
        // DO_Keep: leave this thread's current FTZ/DAZ bits untouched, matching compile_model()'s own behavior.
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
    // Member initializers run before the constructor body, so state is captured before any mutation.
#if OV_CPU_HAS_MXCSR_INTRINSICS
    unsigned int m_mxcsr = _mm_getcsr();
#endif
#if OV_THREAD == OV_THREAD_OMP
    int m_omp_max_threads = omp_get_max_threads();
#endif
};

// Always runs the stage task on the calling thread (unlike the default executor, which may hand
// off to a worker via run_and_wait()) -- required for caller-thread-sync dispatch. Safe from the
// Defer() queue-poisoning issue since make_next_stage_task() already wraps the task in try/catch.
class CallerThreadStageExecutor : public ov::threading::ITaskExecutor {
public:
    CallerThreadStageExecutor(std::shared_ptr<ov::threading::IStreamsExecutor> stream_executor,
                              ov::intel_cpu::Config::DenormalsOptMode denormals_opt_mode)
        : m_stream_executor(std::move(stream_executor)),
          m_denormals_opt_mode(denormals_opt_mode) {}

    void run(ov::threading::Task task) override {
        // Apply denormals handling and restore the caller thread's own OpenMP/MXCSR state afterwards.
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
