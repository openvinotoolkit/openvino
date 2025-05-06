// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/runtime/ipipeline_process.hpp"

#include <memory>

#include "openvino/runtime/iasync_infer_request.hpp"

namespace ov {
// IPipelineProcess
IPipelineProcess::DisableCallbackGuard::DisableCallbackGuard(IPipelineProcess& request)
    : _this{&request},
      m_callback{} {
    _this->swap_callbacks(m_callback);
}

IPipelineProcess::DisableCallbackGuard::~DisableCallbackGuard() {
    _this->swap_callbacks(m_callback);
}

IPipelineProcess::~IPipelineProcess() = default;

// PIpelineProcess
PipelineProcess::PipelineProcess() : PipelineProcess(nullptr) {}

PipelineProcess::PipelineProcess(std::function<void(void)> notify)
    : m_mutex{},
      m_futures{},
      m_promise{},
      m_callback{},
      m_fsm_done_event{std::move(notify)} {}

void PipelineProcess::wait() {
    // Just use the last 'futures' member to wait pipeline completion
    if (auto future = get_last_future(); future.valid()) {
        future.get();
    }
}

bool PipelineProcess::wait_for(const std::chrono::milliseconds& timeout) {
    auto has_result = false;
    // Just use the last 'futures' member to wait pipeline completion
    if (auto future = get_last_future(); future.valid()) {
        if (auto status = future.wait_for(timeout); std::future_status::ready == status) {
            future.get();
            has_result = true;
        }
    }
    return has_result;
}

void PipelineProcess::swap_callbacks(callback_func& other) {
    std::lock_guard<std::mutex> lock{m_mutex};
    std::swap(m_callback, other);
}

std::shared_future<void> PipelineProcess::get_last_future() const {
    std::lock_guard<std::mutex> lock{m_mutex};
    return m_futures.empty() ? std::shared_future<void>{} : m_futures.back();
}

ov::threading::Task PipelineProcess::make_next_stage_task(
    const Pipeline::iterator itStage,
    const Pipeline::iterator itEndStage,
    const std::shared_ptr<ov::threading::ITaskExecutor> callbackExecutor) {
    return std::bind(
        [this, itStage, itEndStage](std::shared_ptr<ov::threading::ITaskExecutor>& callbackExecutor) mutable {
            std::exception_ptr currentException = nullptr;
            auto& thisStage = *itStage;
            auto itNextStage = itStage + 1;
            try {
                auto& stageTask = std::get<TASK>(thisStage);
                OPENVINO_ASSERT(nullptr != stageTask);
                stageTask();
                if (itEndStage != itNextStage) {
                    auto& nextStage = *itNextStage;
                    auto& nextStageExecutor = std::get<EXECUTOR>(nextStage);
                    OPENVINO_ASSERT(nullptr != nextStageExecutor);
                    nextStageExecutor->run(make_next_stage_task(itNextStage, itEndStage, std::move(callbackExecutor)));
                }
            } catch (...) {
                currentException = std::current_exception();
            }

            if ((itEndStage == itNextStage) || (nullptr != currentException)) {
                auto lastStageTask = [this, currentException]() mutable {
                    auto promise = std::move(m_promise);
                    std::function<void(std::exception_ptr)> callback;
                    {
                        if (m_fsm_done_event) {
                            m_fsm_done_event();
                        }
                        std::lock_guard<std::mutex> lock{m_mutex};
                        std::swap(callback, m_callback);
                    }
                    if (callback) {
                        try {
                            callback(currentException);
                        } catch (...) {
                            currentException = std::current_exception();
                        }

                        if (std::lock_guard<std::mutex> lock{m_mutex}; !m_callback) {
                            std::swap(callback, m_callback);
                        }
                    }
                    if (nullptr == currentException) {
                        promise.set_value();
                    } else {
                        promise.set_exception(currentException);
                    }
                };

                if (nullptr == callbackExecutor) {
                    lastStageTask();
                } else {
                    callbackExecutor->run(std::move(lastStageTask));
                }
            }
        },
        std::move(callbackExecutor));
}

void PipelineProcess::stop() {
    Futures futures;
    {
        std::lock_guard<std::mutex> lock{m_mutex};
        m_callback = {};
        futures = std::move(m_futures);
    }
    for (auto&& future : futures) {
        if (future.valid()) {
            future.wait();
        }
    }
};

void PipelineProcess::prepare_sync() {
    prepare_async();
}

void PipelineProcess::prepare_async() {
    std::lock_guard<std::mutex> lock{m_mutex};
    m_futures.erase(std::remove_if(std::begin(m_futures),
                                   std::end(m_futures),
                                   [](auto&& future) {
                                       return future.valid() ? (std::future_status::ready ==
                                                                future.wait_for(std::chrono::milliseconds{0}))
                                                             : true;
                                   }),
                    m_futures.end());
    m_promise = {};
    m_futures.emplace_back(m_promise.get_future().share());
}

PipelineProcess::pipeline_process_func PipelineProcess::sync_pipeline_func() {
    return async_pipeline_func();
};

PipelineProcess::pipeline_process_func PipelineProcess::async_pipeline_func() {
    return [this](const Pipeline::iterator first_stage,
                  const Pipeline::iterator last_stage,
                  const threading::ITaskExecutor::Ptr callback_executor) {
        auto& first_stage_executor = std::get<EXECUTOR>(*first_stage);
        OPENVINO_ASSERT(nullptr != first_stage_executor);
        first_stage_executor->run(make_next_stage_task(first_stage, last_stage, std::move(callback_executor)));
    };
}

void PipelineProcess::set_exception(std::exception_ptr exception) {
    std::lock_guard<std::mutex> lock{m_mutex};
    m_promise.set_exception(exception);
}

void PipelineProcess::set_callback(std::function<void(std::exception_ptr)> callback) {
    m_callback = std::move(callback);
}

PipelineProcess::DisableCallbackGuard PipelineProcess::disable_callback() {
    return DisableCallbackGuard{*this};
}

}  // namespace ov
