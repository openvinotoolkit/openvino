// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <list>
#include <vector>

namespace cldnn {

class stream;
class command_recorder;
class command_list;
class primitive_inst;
class event;

/// @brief Defines the execution mode of a network within a record-replay session.
enum class network_exec_mode {
    immediate,  ///< Execute commands immediately without recording.
    record,     ///< Record commands for later replay.
    replay      ///< Replay previously recorded commands.
};

/// @brief Manages the record and replay of GPU command streams for a network execution session.
class record_replay_session {
public:
    using ptr = std::shared_ptr<record_replay_session>;
    explicit record_replay_session(stream& s);

    /// @brief Begins an iteration of network execution within the record-replay session.
    /// @param deps A list of events that the current iteration depends on.
    /// @param order The execution order of primitives for this iteration.
    /// @return The execution mode used for this iteration.
    network_exec_mode begin_iteration(const std::vector<std::shared_ptr<event>>& deps,
                              const std::list<std::shared_ptr<primitive_inst>>& order);

    /// @brief Ends the current iteration within the record-replay session.
    /// @note Must be called after each iteration with network_exec_mode::record
    void end_iteration();

    /// @brief Invalidates the recorded command list.
    void invalidate();

private:
    stream& _stream;
    command_recorder& _recorder;
    std::shared_ptr<command_list> _cmd_list;
    bool _valid = false;
};

}  // namespace cldnn
