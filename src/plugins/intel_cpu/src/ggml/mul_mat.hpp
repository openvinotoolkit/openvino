// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <cstdint>

namespace ov {
namespace intel_cpu {
void ggml_mul_mat(const int64_t M,
                  const int64_t N,
                  const int64_t K,
                  const float* A_ptr,
                  const float* B_ptr,
                  float* dst_ptr,
                  const float* bias_ptr);
}  // namespace intel_cpu
}  // namespace ov