// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include "cpu/x64/jit_generator.hpp"
#include "cpu_memory.h"

using namespace dnnl;
using namespace dnnl::impl::cpu::x64;
using namespace Xbyak;

namespace ov {
namespace intel_cpu {

struct jit_has_subnormals_base;

class HasSubnormals {
public:
    HasSubnormals();

    bool execute(const IMemory& src);

    std::shared_ptr<jit_has_subnormals_base> kernel;
};

}   // namespace intel_cpu
}   // namespace ov
