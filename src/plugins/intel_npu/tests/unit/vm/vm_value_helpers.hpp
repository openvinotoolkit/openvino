//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vm_runtime/virtual_machine.h"

#include <cstdint>
#include <cstring>

namespace ov {

namespace intel_npu::vm {

inline npu_vm_value makeI64(int64_t v) {
    npu_vm_value val{};
    val.i64 = v;
    return val;
}

inline npu_vm_value makeF32(float v) {
    npu_vm_value val{};
    val.f32 = v;
    return val;
}

inline npu_vm_value makeF64(double v) {
    npu_vm_value val{};
    val.f64 = v;
    return val;
}

inline npu_vm_value makeBuffer(uint8_t* data, uint32_t size) {
    npu_vm_value val{};
    val.buffer.data = data;
    val.buffer.size = size;
    return val;
}

}  // namespace intel_npu::vm

}  // namespace ov
