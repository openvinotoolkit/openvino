// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/runtime/iinfer_request_fsm.hpp"

#include "openvino/runtime/exception.hpp"
#include "openvino/util/variant_visitor.hpp"

namespace ov {

IInferRequestFsm::DisableCallbackGuard::DisableCallbackGuard(IInferRequestFsm& request_fsm)
    : _this{&request_fsm},
      m_callback{} {
    const auto lock = _this->lock();
    std::swap(m_callback, _this->m_callback);
}

IInferRequestFsm::DisableCallbackGuard::~DisableCallbackGuard() {
    const auto lock = _this->lock();
    _this->m_callback = m_callback;
}

IInferRequestFsm::~IInferRequestFsm() = default;

bool IInferRequestFsm::is_busy() const {
    return std::holds_alternative<Busy>(m_state);
}

bool IInferRequestFsm::is_cancelled() const {
    return std::holds_alternative<Cancelled>(m_state);
}

std::unique_lock<std::mutex> IInferRequestFsm::lock() {
    return std::unique_lock<std::mutex>{m_mutex};
}

IInferRequestFsm::DisableCallbackGuard IInferRequestFsm::disable_callback() {
    return DisableCallbackGuard{*this};
}

void IInferRequestFsm::set_callback(std::function<void(std::exception_ptr)> callback) {
    m_callback = std::move(callback);
}

void IInferRequestFsm::stop() {
    std::visit(util::VariantVisitor{[this, ev = StopEvent{}](const Busy& state) {
                                        on_event(state, ev);
                                    },
                                    [this, ev = StopEvent{}](const Idle& state) {
                                        on_event(state, ev);
                                    },
                                    [this](const auto&) -> void {
                                        const auto l = lock();
                                        m_state = Stop{};
                                    }},
               m_state);
}

void IInferRequestFsm::cancel() {
    std::visit(util::VariantVisitor{[this, ev = CancelEvent{}](const Busy& state) {
                                        on_event(state, ev);
                                    },
                                    [](const auto&) {}},
               m_state);
}

void IInferRequestFsm::start(const Pipeline::iterator first_stage,
                             const Pipeline::iterator last_stage,
                             const threading::ITaskExecutor::Ptr callback_executor) {
    const auto event = StartEvent{first_stage, last_stage, std::move(callback_executor)};
    std::visit(util::VariantVisitor{[this, &event](const auto& state) {
                                        on_event(state, event);
                                    },
                                    [](const Stop&) {}},
               m_state);
}

void IInferRequestFsm::on_event(const Busy&, const StopEvent&) {
    {
        auto l = lock();
        m_callback = {};
        m_state = Stop{};
    }
    m_done_signal.notify_all();
}

void IInferRequestFsm::on_event(const Idle&, const StopEvent&) {
    const auto l = lock();
    m_callback = {};
    m_state = Stop{};
}

void IInferRequestFsm::on_event(const Busy&, const CancelEvent&) {
    {
        auto l = lock();
        m_state = Cancelled{};
    }
    m_done_signal.notify_all();
}

void IInferRequestFsm::on_event(const Idle&, const StartEvent&) {
    const auto l = lock();
    m_state = Busy{};
}

void IInferRequestFsm::on_event(const Busy&, const StartEvent&) {
    const auto l = lock();
    ov::Busy::create("Infer Request is busy");
}

void IInferRequestFsm::on_event(const Cancelled&, const StartEvent&) {
    const auto l = lock();
    ov::Cancelled::create("Infer Request was canceled");
}

void IInferRequestFsm::wait() {
    if (auto l = lock(); is_busy()) {
        m_done_signal.wait(l);
    }
}

bool IInferRequestFsm::wait_for(const std::chrono::milliseconds& timeout) {
    if (auto l = lock(); is_busy()) {
        return std::cv_status::no_timeout == m_done_signal.wait_for(l, timeout);
    } else {
        return true;
    }
}

}  // namespace ov
