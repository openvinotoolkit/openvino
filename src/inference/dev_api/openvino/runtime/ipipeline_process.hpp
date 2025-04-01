// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief OpenVINO Runtime interface pipeline processing
 * @file openvino/runtime/ipipeline_process_fsm.hpp
 */

#pragma once

#include <chrono>
#include <functional>
#include <future>
#include <mutex>
#include <vector>

#include "openvino/runtime/threading/itask_executor.hpp"

namespace ov {

class OPENVINO_RUNTIME_API IPipelineProcess {
protected:
    struct DisableCallbackGuard {
        DisableCallbackGuard() = delete;
        DisableCallbackGuard(const DisableCallbackGuard&) = delete;
        DisableCallbackGuard& operator=(const DisableCallbackGuard&) = delete;

        explicit DisableCallbackGuard(IPipelineProcess& pipeline);
        ~DisableCallbackGuard();

        IPipelineProcess* _this;
        std::function<void(std::exception_ptr)> m_callback;
    };

public:
    using Stage = std::pair<threading::ITaskExecutor::Ptr, threading::Task>;
    using Pipeline = std::vector<Stage>;
    using pipeline_process_func = std::function<void(const Pipeline::iterator first_stage,
                                                     const Pipeline::iterator lasts_stage,
                                                     const threading::ITaskExecutor::Ptr callback_executor)>;
    using callback_func = std::function<void(std::exception_ptr)>;

    virtual ~IPipelineProcess();

    virtual void wait() = 0;
    virtual bool wait_for(const std::chrono::milliseconds& timeout) = 0;
    virtual void stop() = 0;
    virtual void prepare_sync() = 0;
    virtual void prepare_async() = 0;
    virtual pipeline_process_func sync_pipeline_func() = 0;
    virtual pipeline_process_func async_pipeline_func() = 0;
    virtual void set_exception(std::exception_ptr) = 0;
    virtual void set_callback(std::function<void(std::exception_ptr)>) = 0;
    virtual DisableCallbackGuard disable_callback() = 0;

protected:
    enum Stage_e : std::uint8_t { EXECUTOR, TASK };
    virtual void swap_callbacks(callback_func& other) = 0;
};

class OPENVINO_RUNTIME_API PipelineProcess : public IPipelineProcess {
public:
    PipelineProcess();
    PipelineProcess(std::function<void(void)> fsm_notify);

    void wait() override;
    bool wait_for(const std::chrono::milliseconds& timeout) override;
    void stop() override;
    void prepare_sync() override;
    void prepare_async() override;
    pipeline_process_func sync_pipeline_func() override;
    pipeline_process_func async_pipeline_func() override;
    void set_exception(std::exception_ptr) override;
    void set_callback(std::function<void(std::exception_ptr)>) override;
    DisableCallbackGuard disable_callback() override;

private:
    void swap_callbacks(callback_func& other) override;
    std::shared_future<void> get_last_future() const;
    ov::threading::Task make_next_stage_task(const Pipeline::iterator itStage,
                                             const Pipeline::iterator itEndStage,
                                             const std::shared_ptr<ov::threading::ITaskExecutor> callbackExecutor);

    using Futures = std::vector<std::shared_future<void>>;
    mutable std::mutex m_mutex;
    Futures m_futures;
    std::promise<void> m_promise;
    std::function<void(std::exception_ptr)> m_callback;  //!< Called on on success or failure of asynchronous request.
    std::function<void()> m_fsm_done_event;              //!< Called when pipeline done to notify FSM.
};
}  // namespace ov
