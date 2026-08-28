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

enum class network_exec_mode {
    immediate,
    record,
    replay
};

/// @brief Manages the record and replay of GPU command streams for a network execution session.
class record_replay_session {
public:
    using ptr = std::shared_ptr<record_replay_session>;
    explicit record_replay_session(stream& s);

    network_exec_mode begin_iteration(const std::vector<std::shared_ptr<event>>& deps,
                              const std::list<std::shared_ptr<primitive_inst>>& order);

    void end_iteration();
    void invalidate();

private:
    stream& _stream;
    command_recorder& _recorder;
    std::shared_ptr<command_list> _cmd_list;
    bool _valid = false;
};

}  // namespace cldnn
