//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vm_export.hpp"

#include <cstdint>

namespace intel_npu::vm {

bool NPU_VM_EXPORT checkedMultiply(uint64_t lhs, uint64_t rhs, uint64_t& result);
bool NPU_VM_EXPORT checkedAddProduct(uint64_t lhs, uint64_t rhs, uint64_t& accumulator);
bool NPU_VM_EXPORT checkedMultiplyNonNegative(int64_t lhs, int64_t rhs, int64_t& result);

}  // namespace intel_npu::vm
