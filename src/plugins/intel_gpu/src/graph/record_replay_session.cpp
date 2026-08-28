// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/graph/record_replay_session.hpp"
#include "intel_gpu/runtime/stream.hpp"
#include "intel_gpu/runtime/event.hpp"
#include "intel_gpu/runtime/command_list.hpp"
#include "intel_gpu/runtime/command_recorder.hpp"

#include "primitive_inst.h"

namespace cldnn {

record_replay_session::record_replay_session(stream& s)
    : _stream(s), _recorder(*s.get_recorder()),
    _cmd_list(_recorder.create_command_list()) {}

network_exec_mode record_replay_session::begin_iteration(const std::vector<std::shared_ptr<event>>& deps,
                              const std::list<std::shared_ptr<primitive_inst>>& order) {
    if (!deps.empty()) {
        _stream.enqueue_marker(deps);
    }
    if (_valid) {
        for (auto& inst : order) {
            inst->reset_out_event();
        }
        _cmd_list->enqueue();
        GPU_DEBUG_TRACE_DETAIL << "[GPU][REC] Replayed last iteration" << std::endl;
        return network_exec_mode::replay;
    }
    _recorder.start_recording(_cmd_list);
    GPU_DEBUG_TRACE_DETAIL << "[GPU][REC] Started recording iteration" << std::endl;
    return network_exec_mode::record;
}

void record_replay_session::end_iteration() {
    auto finished = _recorder.stop_recording();
    _valid = (finished == _cmd_list);
    GPU_DEBUG_TRACE_DETAIL << "[GPU][REC] Stream recording " << (_valid ? "succeeded" : "failed") << std::endl;
}

void record_replay_session::invalidate() { _valid = false; }

}  // namespace cldnn
