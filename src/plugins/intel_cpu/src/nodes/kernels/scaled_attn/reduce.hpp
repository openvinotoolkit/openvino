// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>
#include <openvino/core/type/element_type.hpp>

namespace InferenceEngine {
namespace Extensions {
namespace Cpu {
namespace XARCH {

void attn_reduce(void* dst, float* temp, size_t M, size_t S, size_t temp_stride, ov::element::Type input_precision);

}  // namespace XARCH
}  // namespace Cpu
}  // namespace Extensions
}  // namespace InferenceEngine