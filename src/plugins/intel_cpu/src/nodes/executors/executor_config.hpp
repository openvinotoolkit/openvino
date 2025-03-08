// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "memory_arguments.hpp"
#include "post_ops.hpp"

namespace ov::intel_cpu::executor {

template <typename Attrs>
struct Config {
    MemoryDescArgs descs;
    Attrs attrs;
    PostOps postOps;
};

}  // namespace ov::intel_cpu::executor
