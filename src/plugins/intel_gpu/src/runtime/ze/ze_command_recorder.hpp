// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "intel_gpu/runtime/command_recorder.hpp"

namespace cldnn::ze {
class ze_stream;
class ze_command_recorder : public command_recorder {
public:
    explicit ze_command_recorder(ze_stream& stream);

    command_list::ptr create_command_list() const override;
protected:
    ze_stream& _stream;
};

}  // namespace cldnn::ze
