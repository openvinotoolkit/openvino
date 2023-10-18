// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>
#include "ie_precision.hpp"

namespace InferenceEngine {
namespace Extensions {
namespace Cpu {
namespace XARCH {

void attn_acc_value(float* out, float weight, void* v, size_t S, Precision input_precision);

}  // namespace XARCH
}  // namespace Cpu
}  // namespace Extensions
}  // namespace InferenceEngine