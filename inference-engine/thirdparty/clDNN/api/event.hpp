// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///////////////////////////////////////////////////////////////////////////////////////////////////
#pragma once
#include "cldnn.hpp"
#include "engine.hpp"
#include "profiling.hpp"
#include <algorithm>
#include <cassert>
#include <vector>
#include <memory>
#include <functional>
#include <stdexcept>

namespace cldnn {

/// @addtogroup cpp_api C++ API
/// @{

/// @addtogroup cpp_event Events Support
/// @{

struct event_impl;

/// @brief user-defined event handler callback.
using event_handler = std::function<void(void*)>;

/// @brief Represents an clDNN Event object
struct event {
    /// @brief Create an event which can be set to 'completed' by user.
    static event create_user_event(const engine& engine, uint32_t net_id);

    /// @brief Construct from C API handler @ref ::cldnn_event.
    explicit event(event_impl* impl) : _impl(impl) {
        if (_impl == nullptr) throw std::invalid_argument("implementation pointer should not be null");
    }

    event(const event& other) : _impl(other._impl) {
        retain();
    }

    event& operator=(const event& other) {
        if (_impl == other._impl) return *this;
        release();
        _impl = other._impl;
        retain();
        return *this;
    }

    ~event() {
        release();
    }

    friend bool operator==(const event& lhs, const event& rhs) { return lhs._impl == rhs._impl; }
    friend bool operator!=(const event& lhs, const event& rhs) { return !(lhs == rhs); }

    /// @brief Wait for event completion.
    void wait() const;

    /// @brief Set event status to 'completed'.
    void set() const;

    /// @brief Register call back to be called on event completion.
    void set_event_handler(event_handler handler, void* param) const;

    /// @brief Get profiling info for the event associated with network output.
    std::vector<instrumentation::profiling_interval> get_profiling_info() const;

    /// @brief Returns C API event handler.
    event_impl* get() const { return _impl; }

private:
    event_impl* _impl;
    void retain();
    void release();
};
CLDNN_API_CLASS(event)

/// @}
/// @}
}  // namespace cldnn
