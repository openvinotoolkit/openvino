// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "frontend_manager/extension.hpp"

ov::frontend::TelemetryExtension::TelemetryExtension(std::string event_category,
                                                     ov::frontend::TelemetryExtension::event_callback& send_event,
                                                     ov::frontend::TelemetryExtension::error_callback& send_error,
                                                     ov::frontend::TelemetryExtension::error_callback& send_stack_trace)
    : m_event_category(event_category),
      m_send_event(send_event),
      m_send_error(send_error),
      m_send_stack_trace(send_stack_trace) {}

void ov::frontend::TelemetryExtension::send_event(const std::string& action, const std::string& label, int value) {
    if (m_send_event) {
        m_send_event(m_event_category, action, label, value);
    }
}

void ov::frontend::TelemetryExtension::send_error(const std::string& error_message) {
    if (m_send_error) {
        m_send_error(m_event_category, error_message);
    }
}

void ov::frontend::TelemetryExtension::send_stack_trace(const std::string& error_message) {
    if (m_send_stack_trace) {
        m_send_stack_trace(m_event_category, error_message);
    }
}
