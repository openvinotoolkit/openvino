// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <fstream>
#include <iostream>
#include <type_traits>

#include "frontend_manager_defs.hpp"
#include "openvino/core/extension.hpp"
#include "openvino/pass/graph_rewrite.hpp"
#include "openvino/pass/manager.hpp"
#include "openvino/pass/pass.hpp"

namespace ov {
namespace frontend {

/// \brief Provides callback to report telemetry information back to Python code
class FRONTEND_API TelemetryExtension : public ov::Extension {
public:
    using session_callback = const std::function<void(const std::string& category)>;
    using error_callback = const std::function<void(const std::string& category, const std::string& error_message)>;
    using event_callback = const std::function<
        void(const std::string& category, const std::string& action, const std::string& label, int value)>;
    using shutdown_callback = const std::function<void(float timeout)>;
    TelemetryExtension(event_callback& send_event,
                       error_callback& send_error,
                       session_callback& start_session,
                       session_callback& end_session,
                       shutdown_callback& force_shutdown,
                       error_callback& send_stack_trace)
        : m_send_event(send_event),
          m_send_error(send_error),
          m_start_session(start_session),
          m_end_session(end_session),
          m_force_shutdown(force_shutdown),
          m_send_stack_trace(send_stack_trace) {
    }

    void send_event(const std::string& category, const std::string& action, const std::string& label, int value = 1) {
        if (m_send_event) {
            std::cout << "XXXXXXXXX NGRAPH TelemetryExtenxion send_event" << std::endl;
            m_send_event(category, action, label, value);
        }
    }

    void send_error(const std::string& category, const std::string& error_message) {
        if (m_send_error) {
            m_send_error(category, error_message);
        }
    }

    void start_session(const std::string& category) {
        if (m_start_session) {
            m_start_session(category);
        }
    }

    void end_session(const std::string& category) {
        if (m_end_session) {
            m_end_session(category);
        }
    }

    void force_shutdown(float timeout = 1.0f) {
        if (m_force_shutdown) {
            m_force_shutdown(timeout);
        }
    }
    void send_stack_trace(const std::string& category, const std::string& error_message) {
        if (m_send_stack_trace) {
            m_send_stack_trace(category, error_message);
        }
    }

private:
    event_callback m_send_event;
    error_callback m_send_error;
    error_callback m_send_stack_trace;
    session_callback m_start_session;
    session_callback m_end_session;
    shutdown_callback m_force_shutdown;
};

}  // namespace frontend
}  // namespace ov
