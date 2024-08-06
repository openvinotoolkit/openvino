// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include "cpu_memory.h"

namespace ov {
namespace intel_cpu {

struct jit_has_subnormals_base;

class HasSubnormals {
public:
    bool execute(const IMemory& src);
};

}   // namespace intel_cpu
}   // namespace ov
