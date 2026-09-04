// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/core/except.hpp"

#include <memory>

namespace cldnn {
enum class command_list_status {
    open = 0,
    closed = 1,
    enqueued = 2,
};

/// @brief Commands that can be recorded and executed
class command_list {
public:
    using ptr = std::shared_ptr<command_list>;
    virtual ~command_list() = default;

    command_list_status get_status() const { return _status; }

    /// @brief Reset command list to initial empty state.
    void reset() {
        if (_status == command_list_status::enqueued) {
            wait();
        }
        reset_impl();
        _status = command_list_status::open;
    }

    /// @brief Close command list and make it ready for execution.
    void close() {
        OPENVINO_ASSERT(_status == command_list_status::open, "[GPU] Can't close command list that is not open");
        close_impl();
        _status = command_list_status::closed;
    }

    /// @brief Enqueue command list on associated stream.
    /// @note Resources referenced by the command list must remain valid during command list execution.
    void enqueue() {
        OPENVINO_ASSERT(_status != command_list_status::open, "[GPU] Can't enqueue command list that is open");
        if (_status == command_list_status::enqueued) {
            wait();
        }
        enqueue_impl();
        _status = command_list_status::enqueued; 
    }

    /// @brief Wait for command list execution to finish.
    void wait() {
        OPENVINO_ASSERT(_status != command_list_status::open, "[GPU] Can't wait on command list that is open");
        if (_status == command_list_status::closed) {
            return;
        }
        wait_impl();
        _status = command_list_status::closed; 
    }
protected:
    virtual void reset_impl() = 0;
    virtual void close_impl() = 0;
    virtual void enqueue_impl() = 0;
    virtual void wait_impl() = 0;

    command_list_status _status = command_list_status::open;
};

}  // namespace cldnn
