// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <oneapi/dnnl/dnnl.hpp>

#include "dnnl_aliases.hpp"
#include "nodes/executors/memory_arguments.hpp"

namespace ov::intel_cpu {

struct DnnlPrimitiveAttrs {
    dnnl::primitive_attr attr;
    dnnl_primitive_args dnnlArgs;
    MemoryArgs cpuArgs;
    bool legacyZeroPoints;
};

}  // namespace ov::intel_cpu
