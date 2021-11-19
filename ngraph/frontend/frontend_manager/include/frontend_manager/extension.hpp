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
    using error_callback = const std::function<void(const std::string& category, const std::string& error_message)>;
    using event_callback = const std::function<
        void(const std::string& category, const std::string& action, const std::string& label, int value)>;
    TelemetryExtension(std::string event_category,
                       event_callback& send_event,
                       error_callback& send_error,
                       error_callback& send_stack_trace)
        : m_event_category(event_category),
          m_send_event(send_event),
          m_send_error(send_error),
          m_send_stack_trace(send_stack_trace) {}

    void send_event(const std::string& action, const std::string& label, int value = 1) {
        if (m_send_event) {
            m_send_event(m_event_category, action, label, value);
        }
    }

    void send_error(const std::string& error_message) {
        if (m_send_error) {
            m_send_error(m_event_category, error_message);
        }
    }

    void send_stack_trace(const std::string& error_message) {
        if (m_send_stack_trace) {
            m_send_stack_trace(m_event_category, error_message);
        }
    }

private:
    std::string m_event_category;
    event_callback m_send_event;
    error_callback m_send_error;
    error_callback m_send_stack_trace;
};

}  // namespace frontend
}  // namespace ov
