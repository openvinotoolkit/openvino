// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "intel_gpu/runtime/debug_configuration.hpp"

#include <memory>
#include <list>
#include <vector>

namespace cldnn {

class stream;
class command_recorder;
class command_list;
class primitive_inst;
struct event;

/// @brief Manages the record and replay of GPU command streams for a network execution session.
class record_replay_session {
public:
    using ptr = std::shared_ptr<record_replay_session>;
    explicit record_replay_session(stream& s);

    /// @brief Start recording commands on the associated stream.
    /// @param deps Events that must complete before recorded commands are executed.
    /// @note Events from deps are not added to the command list.
    /// @note For each begin_recording() there must be corresponding call to end_recording().
    void begin_recording(const std::vector<std::shared_ptr<event>>& deps);

    /// @brief Ends the current recording session and executes recorded commands.
    void end_recording();

    /// @brief Attempt to replay previosuly recorded command list.
    /// @param deps Events that must complete before commands are replayed.
    /// @param primitives A list of primitives to replay. Must be the same as the list used for the recording.
    /// @return True if replay was successful, false otherwise.
    bool replay(const std::vector<std::shared_ptr<event>>& deps,
                const std::list<std::shared_ptr<primitive_inst>>& primitives);

    /// @brief Invalidates the recorded command list.
    void invalidate() {
        _valid = false;
        GPU_DEBUG_TRACE_DETAIL << "[REC] Recorded command list was invalidated";
    }

private:
    stream& _stream;
    command_recorder& _recorder;
    std::shared_ptr<command_list> _cmd_list;
    bool _valid = false;
};

}  // namespace cldnn
