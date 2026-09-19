// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/graph/record_replay_session.hpp"
#include "intel_gpu/runtime/stream.hpp"
#include "intel_gpu/runtime/event.hpp"
#include "intel_gpu/runtime/command_list.hpp"
#include "intel_gpu/runtime/command_recorder.hpp"
#include "intel_gpu/runtime/debug_configuration.hpp"

#include "primitive_inst.h"

namespace cldnn {

record_replay_session::record_replay_session(stream& s)
    : _stream(s), _recorder(*s.get_recorder()),
    _cmd_list(_recorder.create_command_list()) {}

void record_replay_session::begin_recording(const std::vector<std::shared_ptr<event>>& deps) {
    if (!deps.empty()) {
        // Ensure dependencies complete before recording
        _stream.enqueue_barrier(deps);
    }
    if (_cmd_list->get_status() != command_list_status::open) {
        _cmd_list->reset();
    }
    _recorder.start_recording(_cmd_list);
    GPU_DEBUG_TRACE_DETAIL << "[REC] Started recording command list" << std::endl;
}

void record_replay_session::end_recording() {
    auto finished = _recorder.stop_recording();
    _valid = (finished == _cmd_list);
    GPU_DEBUG_TRACE_DETAIL << "[REC] Command list recording " << (_valid ? "succeeded" : "failed") << std::endl;
}

bool record_replay_session::replay(const std::vector<std::shared_ptr<event>>& deps, const std::list<std::shared_ptr<primitive_inst>>& primitives) {
    if (!_valid) {
        return false;
    }
    if (!deps.empty()) {
        // Ensure dependencies complete before replaying
        _stream.enqueue_barrier(deps);
    }
    _cmd_list->wait();
    for (const auto& inst : primitives) {
        inst->reset_out_event();
    }
    _cmd_list->enqueue();
    GPU_DEBUG_TRACE_DETAIL << "[REC] Replayed command list" << std::endl;

    return true;
}

}  // namespace cldnn
