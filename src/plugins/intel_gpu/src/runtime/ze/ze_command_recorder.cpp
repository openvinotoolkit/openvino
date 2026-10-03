// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ze_command_recorder.hpp"
#include "ze_stream.hpp"
#include "ze_command_list.hpp"

namespace cldnn::ze {
    
ze_command_recorder::ze_command_recorder(ze_stream& stream)
    : _stream(stream) {}

command_list::ptr ze_command_recorder::create_command_list() const {
    return std::make_shared<ze_command_list>(_stream, _stream.get_queue_type());
}

}  // namespace cldnn::ze
