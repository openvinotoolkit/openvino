// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief OpenVINO Runtime InfeR interface
 * @file openvino/runtime/iinfer_request_fsm.hpp
 */

#pragma once

#include <chrono>
#include <condition_variable>
#include <mutex>
#include <variant>
#include <vector>

#include "openvino/runtime/threading/itask_executor.hpp"

namespace ov {

class OPENVINO_RUNTIME_API IInferRequestFsm {
    /**
     * @brief The DisableCallbackGuard class is used to disable the callback until the guard is destroyed.
     */
    struct DisableCallbackGuard {
        DisableCallbackGuard() = delete;
        DisableCallbackGuard(const DisableCallbackGuard&) = delete;
        DisableCallbackGuard& operator=(const DisableCallbackGuard&) = delete;

        explicit DisableCallbackGuard(IInferRequestFsm& request_fsm);
        ~DisableCallbackGuard();

        IInferRequestFsm* _this;
        std::function<void(std::exception_ptr)> m_callback;
    };

public:
    using Stage = std::pair<threading::ITaskExecutor::Ptr, threading::Task>;
    using Pipeline = std::vector<Stage>;

    virtual ~IInferRequestFsm();

    /**
     * @brief Creates start event with pipline stages to be processed.
     *
     * @param first_stage  Iterator to the first stage of the pipeline.
     * @param last_stage  Iterator to the last stage of the pipeline.
     * @param callback_executor
     */
    virtual void start(const Pipeline::iterator first_stage,
                       const Pipeline::iterator last_stage,
                       const threading::ITaskExecutor::Ptr callback_executor = {});
    /**
     * @brief Create and process the stop event.
     */
    virtual void stop();

    /**
     * @brief Creates and processes the cancel event.
     */
    virtual void cancel();

    /**
     * @brief Waits for the result to become available.
     */
    virtual void wait();

    /**
     * @brief Waits for the result to become available. Blocks until specified timeout has elapsed or the result
     * becomes available, whichever comes first.
     * @param timeout - maximum duration in milliseconds to block for
     * @return A true if results are ready.
     */
    virtual bool wait_for(const std::chrono::milliseconds& timeout);

    /**
     * @brief Checks if the FSM is in busy state.
     * @return True if in busy state, false otherwise.
     */
    bool is_busy() const;

    /**
     * @brief Checks if the FSM is in cancelled state.
     * @return True if in cancelled state, false otherwise.
     */
    bool is_cancelled() const;

    /**
     * @brief Set the callback.
     * @param callback Callback functor to set.
     */
    void set_callback(std::function<void(std::exception_ptr)> callback);

    /**
     * @brief Creates a guard that disables the callback.
     * @return Returns the guard that disables the callback until the guard is destroyed.
     */
    DisableCallbackGuard disable_callback();

protected:
    std::unique_lock<std::mutex> lock();

    // defined base states
    struct Idle {};
    struct Busy {};
    struct Cancelled {};
    struct Stop {};

    using State = std::variant<Idle, Busy, Cancelled, Stop>;

    // define base events
    struct StartEvent {
        const Pipeline::iterator first_stage;
        const Pipeline::iterator last_stage;
        const threading::ITaskExecutor::Ptr callback_executor;
    };
    struct StopEvent {};
    struct CancelEvent {};

    using Event = std::variant<StartEvent, StopEvent, CancelEvent>;

    // predefined event for base state machine transitions
    // Handle the Stop event
    virtual void on_event(const Busy&, const StopEvent&);
    virtual void on_event(const Idle&, const StopEvent&);
    // Handle the Start event
    virtual void on_event(const Idle&, const StartEvent&);
    virtual void on_event(const Busy&, const StartEvent&);
    virtual void on_event(const Cancelled&, const StartEvent&);
    // Handle the Cancel event
    virtual void on_event(const Busy&, const CancelEvent&);

    State m_state{};                                       //!< State of the request.
    std::function<void(std::exception_ptr)> m_callback{};  //!< Called on on success or failure of asynchronous request.
    mutable std::mutex m_mutex{};                          //!< Mutex to protect state and callback.
    std::condition_variable m_done_signal{};               //!< Used to notify when the job is done.
};
}  // namespace ov
