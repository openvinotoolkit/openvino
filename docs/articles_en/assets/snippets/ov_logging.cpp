// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <openvino/core/log.hpp>
#include <openvino/runtime/core.hpp>

#include <functional>
#include <iostream>
#include <mutex>
#include <string>
#include <string_view>

void part0() {
//! [ov:logging:part0]
// Set a custom log callback to capture OpenVINO log messages
const std::function<void(std::string_view)> log_callback{[](std::string_view msg) {
    std::cout << "[OpenVINO] " << msg;
}};
ov::util::set_log_callback(log_callback);

// ... perform OpenVINO operations (compile_model, infer, etc.) ...
// All log messages from OpenVINO will be forwarded to the callback above.

// Reset the callback to restore default logging to std::cout
ov::util::reset_log_callback();
//! [ov:logging:part0]
}

void part1() {
//! [ov:logging:part1]
// Disable all OpenVINO log output by passing an empty callable
ov::util::set_log_callback({});

// ... perform OpenVINO operations silently ...

// Re-enable default logging
ov::util::reset_log_callback();
//! [ov:logging:part1]
}

void part2() {
//! [ov:logging:part2]
// Thread-safe logging: collect messages into a string buffer
std::mutex log_mutex;
std::string log_buffer;
const std::function<void(std::string_view)> log_callback{[&](std::string_view msg) {
    std::lock_guard<std::mutex> lock(log_mutex);
    log_buffer.append(msg);
    log_buffer.push_back('\n');
}};

ov::util::set_log_callback(log_callback);

ov::Core core;
// ... operations that produce log output ...

ov::util::reset_log_callback();

// Process collected log messages
std::cout << "Captured log:\n" << log_buffer;
//! [ov:logging:part2]
}

int main() {
    try {
        part0();
        part1();
        part2();
    } catch (...) {
    }
    return 0;
}
