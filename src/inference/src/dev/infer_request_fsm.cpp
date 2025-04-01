// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/runtime/infer_request_fsm.hpp"

#include "openvino/runtime/exception.hpp"
#include "openvino/util/variant_visitor.hpp"

namespace ov {

InferRequestFsm::InferRequestFsm() : m_state{}, m_mutex{} {
    // Set this FSM to Idle state
    std::get<Idle>(m_state).m_fsm = this;
}

bool InferRequestFsm::is_ready() const {
    return std::holds_alternative<Idle>(m_state);
}

bool InferRequestFsm::is_busy() const {
    return std::holds_alternative<Busy>(m_state);
}

bool InferRequestFsm::is_cancelled() const {
    return std::holds_alternative<Cancelled>(m_state);
}

std::unique_lock<std::mutex> InferRequestFsm::lock() {
    return std::unique_lock<std::mutex>{m_mutex};
}

// Idle
void InferRequestFsm::Idle::on_event(const StartEvent& event) {
    try {
        {
            auto l = m_fsm->lock();
            m_fsm->m_state = Busy{m_fsm};
        }
        if (event.process_pipeline) {
            event.process_pipeline(event.first_stage, event.last_stage, event.callback_executor);
        } else {
            auto l = m_fsm->lock();
            m_fsm->m_state = Idle{m_fsm};
        }
    } catch (...) {
        auto l = m_fsm->lock();
        m_fsm->m_state = Idle{m_fsm};
        throw;
    }
}

void InferRequestFsm::Idle::on_event(const StopEvent& event) {
    auto l = m_fsm->lock();
    m_fsm->m_state = Stop{};
}

void InferRequestFsm::Busy::on_event(const StartEvent&) {
    ov::Busy::create("Infer Request is busy");
}

void InferRequestFsm::Busy::on_event(const StopEvent&) {
    auto l = m_fsm->lock();
    m_fsm->m_state = Stop{};
}

void InferRequestFsm::Busy::on_event(const CancelEvent&) {
    const auto l = m_fsm->lock();
    m_fsm->m_state = Cancelled{m_fsm};
}

void InferRequestFsm::Busy::on_event(const DoneEvent&) {
    const auto l = m_fsm->lock();
    m_fsm->m_state = Idle{m_fsm};
}

void InferRequestFsm::Cancelled::on_event(const StartEvent&) {
    ov::Cancelled::create("Infer Request was canceled");
}

void InferRequestFsm::Cancelled::on_event(const StopEvent&) {
    auto l = m_fsm->lock();
    m_fsm->m_state = Stop{};
}

void InferRequestFsm::Cancelled::on_event(const DoneEvent&) {
    const auto l = m_fsm->lock();
    m_fsm->m_state = Idle{m_fsm};
}
}  // namespace ov
