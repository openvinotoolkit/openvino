// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "memory_arguments.hpp"

namespace ov::intel_cpu::executor {

template <typename Attrs>
struct Config {
    MemoryDescArgs descs;
    Attrs attrs;
};

}  // namespace ov::intel_cpu::executor
