// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "async_infer_request.h"

#include <memory>
#include <vector>

#include "openvino/core/except.hpp"
#include "openvino/runtime/iasync_infer_request.hpp"
#include "openvino/runtime/iinfer_request.hpp"
#include "openvino/runtime/threading/istreams_executor.hpp"
#include "openvino/runtime/threading/itask_executor.hpp"

namespace {
thread_local bool g_in_cpu_async_infer_pipeline = false;

class FlaggedTaskExecutor final : public ov::threading::ITaskExecutor {
public:
    explicit FlaggedTaskExecutor(std::shared_ptr<ov::threading::ITaskExecutor> inner)
        : m_inner(std::move(inner)) {}

    void run(ov::threading::Task task) override {
        OPENVINO_ASSERT(m_inner, "FlaggedTaskExecutor has null inner executor");
        auto wrapped = [task = std::move(task)]() mutable {
            g_in_cpu_async_infer_pipeline = true;
            auto reset_flag = []() { g_in_cpu_async_infer_pipeline = false; };
            try {
                task();
            } catch (...) {
                reset_flag();
                throw;
            }
            reset_flag();
        };
        m_inner->run(std::move(wrapped));
    }

private:
    std::shared_ptr<ov::threading::ITaskExecutor> m_inner;
};

std::shared_ptr<ov::threading::ITaskExecutor> wrap_executor(
    const std::shared_ptr<ov::threading::ITaskExecutor>& executor) {
    if (!executor) {
        return executor;
    }
    if (std::dynamic_pointer_cast<FlaggedTaskExecutor>(executor)) {
        return executor;
    }
    return std::make_shared<FlaggedTaskExecutor>(executor);
}
}  // namespace

ov::intel_cpu::AsyncInferRequest::AsyncInferRequest(
    const std::shared_ptr<IInferRequest>& request,
    const std::shared_ptr<ov::threading::ITaskExecutor>& task_executor,
    const std::shared_ptr<ov::threading::ITaskExecutor>& callback_executor,
    const bool is_optimized_single_stream)
    : ov::IAsyncInferRequest(request, wrap_executor(task_executor), wrap_executor(callback_executor)),
      m_internal_request(request) {
    static_cast<SyncInferRequest*>(request.get())->set_async_request(this);
    m_stream_executor = std::dynamic_pointer_cast<ov::threading::IStreamsExecutor>(task_executor);
    m_infer_func = [this]() {
        ov::IAsyncInferRequest::infer();
    };
    if (is_optimized_single_stream) {
        m_infer_func = [this]() {
            check_tensors();
            m_stream_executor->execute([this]() {
                m_internal_request->infer();
            });
        };
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

void ov::intel_cpu::AsyncInferRequest::infer() {
    m_infer_func();
}

void ov::intel_cpu::AsyncInferRequest::cancel() {
    ov::IAsyncInferRequest::cancel();

    if (g_in_cpu_async_infer_pipeline) {
        return;
    }

    try {
        wait();
    } catch (const ov::Cancelled&) {
        return;
    } catch (const std::exception& ex) {
        OPENVINO_ASSERT(false, "Cancel wait failed: ", ex.what());
    } catch (...) {
        OPENVINO_ASSERT(false, "Cancel wait failed with unknown exception");
    }
}
