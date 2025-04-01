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
#include "openvino/runtime/ipipeline_process.hpp"

namespace ov {

class OPENVINO_RUNTIME_API InferRequestFsm {
public:
    using PipelineIter = IPipelineProcess::Pipeline::iterator;

    InferRequestFsm();

    // define base events
    struct StartEvent {
        const PipelineIter first_stage;                         // !< Iterator to the first stage of the pipeline.
        const PipelineIter last_stage;                          // !< Iterator to the last stage of the pipeline.
        const threading::ITaskExecutor::Ptr callback_executor;  // !< Executor for the callback.
        const IPipelineProcess::pipeline_process_func process_pipeline;  // !< Function to process the pipeline.
    };

    struct StopEvent {};
    struct CancelEvent {};
    struct DoneEvent {};

    template <class Event>
    void on_event(const Event& event) {
        std::visit(
            [&event](auto& state) {
                state.on_event(event);
            },
            m_state);
    }

    /**
     * @brief Checks if the FSM is in Idle state.
     * @return True if in Idle state, false otherwise.
     */
    bool is_ready() const;
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
     * @brief Locks the FSM mutex.
     * @return A unique lock for the mutex.
     */
    std::unique_lock<std::mutex> lock();

private:
    template <class... Actions>
    struct EventHandlers : Actions... {
        using Actions::on_event...;
    };

    struct NoAction {
        template <class Event>
        void on_event(const Event&) {}
    };

    using Event = std::variant<StartEvent, StopEvent, CancelEvent>;

    struct StateBase : EventHandlers<NoAction> {
        using EventHandlers::on_event;

        StateBase() : m_fsm{} {};
        StateBase(InferRequestFsm* fsm) : m_fsm{fsm} {}

        InferRequestFsm* m_fsm;
    };

    // defined base states
    struct Idle : public StateBase {
        using StateBase::on_event;

        Idle(InferRequestFsm* fsm) : StateBase{fsm} {}
        Idle() = default;

        void on_event(const StartEvent& event);
        void on_event(const StopEvent& event);
    };

    struct Busy : public StateBase {
        using StateBase::on_event;

        Busy(InferRequestFsm* fsm) : StateBase{fsm} {}

        void on_event(const StartEvent& event);
        void on_event(const StopEvent& event);
        void on_event(const CancelEvent& event);
        void on_event(const DoneEvent& event);
    };

    struct Cancelled : public StateBase {
        using StateBase::on_event;

        Cancelled(InferRequestFsm* fsm) : StateBase{fsm} {}

        void on_event(const StartEvent& event);
        void on_event(const StopEvent& event);
    };

    struct Stop : public EventHandlers<NoAction> {
        using EventHandlers::on_event;
    };

    using State = std::variant<Idle, Busy, Cancelled, Stop>;

    State m_state{};                                       //!< State of the request.
    mutable std::mutex m_mutex{};                          //!< Mutex to protect state and callback.
};
}  // namespace ov
