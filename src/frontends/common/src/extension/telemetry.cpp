// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/extension/telemetry.hpp"

using namespace ov::frontend;

ov::frontend::TelemetryExtension::TelemetryExtension(const std::string& event_category,
                                                     const TelemetryExtension::event_callback& send_event,
                                                     const TelemetryExtension::error_callback& send_error,
                                                     const TelemetryExtension::error_callback& send_stack_trace)
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
