// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include "openvino/core/type/element_type.hpp"

#include <immintrin.h>

namespace ov {
namespace Extensions {
namespace Cpu {
namespace XARCH {

void allreduce_float32(const float* send_buf,
                       float* recv_buf,
                       size_t count);

__m512 cvt_bf16_to_fp32(ov::bfloat16* data);
__m256i cvt_fp32_to_bf16(__m512 data);

void allreduce_bfloat16(ov::bfloat16* send_buf,
                        ov::bfloat16* recv_buf,
                        size_t count);

}  // namespace XARCH
}  // namespace Cpu
}  // namespace Extensions
}  // namespace ov
