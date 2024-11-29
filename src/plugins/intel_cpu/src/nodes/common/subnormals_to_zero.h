// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <cstddef>

namespace ov {
namespace intel_cpu {

void setSubnormalsToZero(float* data, size_t size);

}   // namespace intel_cpu
}   // namespace ov
