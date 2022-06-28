// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/util/shared_object.hpp"

using namespace ov::util;

constexpr auto close_delay = std::chrono::milliseconds(1);
constexpr auto close_check_interval = std::chrono::milliseconds(50);

int SharedObjectDeferCloser::m_count = 0;
std::mutex SharedObjectDeferCloser::m_count_mutex;
std::once_flag SharedObjectDeferCloser::m_atexit_register;

SharedObjectDeferCloser::SharedObjectDeferCloser() {
    std::call_once(m_atexit_register, wait_all_closed);
}

void SharedObjectDeferCloser::operator()(void* shared_object) const {
    std::lock_guard<std::mutex> guard(m_count_mutex);
    std::thread th_lib_close(lib_delayed_close, shared_object);
    th_lib_close.detach();
    ++m_count;
}

inline void SharedObjectDeferCloser::lib_delayed_close(void* shared_object) {
    std::this_thread::sleep_for(close_delay);
    std::lock_guard<std::mutex> guard(m_count_mutex);
    SharedObjectCloser()(shared_object);
    --m_count;
}

inline bool SharedObjectDeferCloser::any_open() {
    std::lock_guard<std::mutex> guard(m_count_mutex);
    return m_count > 0;
}

inline void SharedObjectDeferCloser::wait_all_closed() {
    std::atexit([]() {
        while (any_open()) {
            std::this_thread::sleep_for(close_check_interval);
        }
    });
}
