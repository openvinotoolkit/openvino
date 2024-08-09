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

#if defined(OPENVINO_ARCH_ARM64) && defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
void attn_softmax(ov::float16* a,
                  void* a_dst,
                  float scale,
                  ov::float16* alibi,
                  void* attn_mask,
                  uint8_t* causal_mask,
                  bool select_nfltmax_at_0,
                  size_t len,
                  size_t total_size,
                  ov::element::Type attn_mask_prec,
                  ov::element::Type dst_precision);
#endif
}  // namespace XARCH
}  // namespace Cpu
}  // namespace Extensions
}  // namespace ov