// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>

namespace cldnn {
/// @brief Commands that can be recorded and executed
class command_list {
public:
    using ptr = std::shared_ptr<command_list>;
    virtual ~command_list() = default;
};

}  // namespace cldnn
