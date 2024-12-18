// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <array>
#include <cstddef>
#include <vector>

#include "openvino/core/type/bfloat16.hpp"
#include "openvino/core/type/float16.hpp"

namespace ov {
namespace intel_cpu {

void dynPruneLinear_f16(const float* input,
                       float threshold,
                       float zero_point,
                       const ov::float16* W,
                       float* output,
                       int M,
                       int IC,
                       int OC);

void dynPruneLinear_i8(const float* input,
                       float threshold,
                       float zero_point,
                       const uint8_t* W,
                       const uint8_t* zp,
                       const float* scales,
                       float* output,
                       int M,
                       int IC,
                       int OC);
void dynPruneLinear_i4(const float* input,
                       float threshold,
                       float zero_point,
                       const uint8_t* W,
                       const uint8_t* zp,
                       const float* scales,
                       float* output,
                       int M,
                       int IC,
                       int OC,
                       int IC_group_size);
void dynPruneLinear_repack_i4(uint8_t * src, uint8_t * dst, int IC, int OC);

}  // namespace intel_cpu
}  // namespace ov
