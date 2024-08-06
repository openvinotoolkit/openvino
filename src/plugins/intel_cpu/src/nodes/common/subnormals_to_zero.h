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

void setSubnormalsToZero(float* data, size_t size);

}   // namespace intel_cpu
}   // namespace ov
