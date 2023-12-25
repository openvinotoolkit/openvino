// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include <float.h>

#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>
#include <type_traits>

#if defined(HAVE_AVX2) || defined(HAVE_AVX512F)
#    include <immintrin.h>
#endif

#include "openvino/core/type/bfloat16.hpp"
#include "softmax.hpp"
#include "softmax_kernel.hpp"
#include "common.hpp"

namespace ov {
namespace Extensions {
namespace Cpu {
namespace XARCH {

void attn_softmax(float* a,
                  void* a_dst,
                  float scale,
                  float* alibi,
                  float* attn_mask,
                  uint8_t* causal_mask,
                  bool select_nfltmax_at_0,
                  size_t len,
                  size_t total_size,
                  ov::element::Type dst_precision) {
    attn_softmax_kernel(a, a_dst, scale, alibi, attn_mask, causal_mask, select_nfltmax_at_0, len, total_size, dst_precision);
}

}  // namespace XARCH
}  // namespace Cpu
}  // namespace Extensions
}  // namespace ov