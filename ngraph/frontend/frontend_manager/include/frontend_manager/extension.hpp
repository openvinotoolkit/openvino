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
    using callback = const std::function<void(const std::string& message)>;
    TelemetryExtension(callback& send_event,
                       callback& send_error,
                       callback& start_session,
                       callback& end_session,
                       callback& force_shutdown,
                       callback& send_stack_trace)
        : m_send_event(send_event),
          m_send_error(send_error),
          m_start_session(start_session),
          m_end_session(end_session),
          m_force_shutdown(force_shutdown),
          m_send_stack_trace(send_stack_trace) {}

    void send_event(const std::string& message) {
        m_send_event(message);
    }
    void send_error(const std::string& message) {
        m_send_error(message);
    }
    void start_session(const std::string& message) {
        m_start_session(message);
    }
    void end_session(const std::string& message) {
        m_end_session(message);
    }
    void force_shutdown(const std::string& message) {
        m_force_shutdown(message);
    }
    void send_stack_trace(const std::string& message) {
        m_send_stack_trace(message);
    }

private:
    callback m_send_event;
    callback m_send_error;
    callback m_start_session;
    callback m_end_session;
    callback m_force_shutdown;
    callback m_send_stack_trace;
};

}  // namespace frontend
}  // namespace ov
