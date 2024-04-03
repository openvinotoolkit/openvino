// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>
#include <openvino/core/type/element_type.hpp>

namespace ov {
namespace Extensions {
namespace Cpu {
namespace XARCH {

void reduce_add(char* data_in_bytes, const size_t data_stride, const char* update_in_bytes, const size_t update_stride, const size_t size);

}  // namespace XARCH
}  // namespace Cpu
}  // namespace Extensions
}  // namespace ov
