// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstdint>

namespace ov {
namespace intel_gna {
namespace memory {

int32_t MemoryOffset(void *ptr_target, void *ptr_base);

}  // namespace memory
}  // namespace intel_gna
}  // namespace ov
