// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "command_list.hpp"
#include "openvino/core/except.hpp"

#include <memory>
#include <mutex>

namespace cldnn {
/// @brief Interface for recording commands on associated stream.
class command_recorder {
public:
    using ptr = std::shared_ptr<command_recorder>;
    command_recorder() = default;
    virtual ~command_recorder() = default;
    // Delete copy ctor to prevent sharing active command list
    command_recorder(const command_recorder& other) = delete;

    /// @brief Create command list for recording executed commands.
    /// @return Command list object.
    virtual command_list::ptr create_command_list() const = 0;

    /// @brief Start recording operations executed on the associated stream.
    /// Executed commands are not submitted to the device during recording.
    /// @param cmd_list Command list to record executed commands.
    void start_recording(command_list::ptr cmd_list) {
        std::lock_guard<std::mutex> lock(_mutex);
        OPENVINO_ASSERT(_active_cmd_list == nullptr, "[GPU] Can't start recording while another recording is in progress");
        cmd_list->reset();
        _active_cmd_list = cmd_list;
    }
    /// @brief Get command list that is currently being recorded.
    /// @return Command list that is currently being recorded or nullptr.
    command_list::ptr get_active_command_list() const {
        std::lock_guard<std::mutex> lock(_mutex);
        return _active_cmd_list;
    }

    /// @brief Stop recording and submit all recorded commands to the device.
    /// @return Command list with recorded commands or nullptr if stream was not recording.
    command_list::ptr stop_recording() {
        std::lock_guard<std::mutex> lock(_mutex);
        command_list::ptr ret = nullptr;
        _active_cmd_list.swap(ret);
        if (ret != nullptr) {
            ret->close();
            ret->enqueue();
        }
        return ret;
    }

protected:
    command_list::ptr _active_cmd_list;
    mutable std::mutex _mutex;
};

}  // namespace cldnn
