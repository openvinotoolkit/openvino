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

void attn_softmax(void* a,
                  void* a_dst,
                  float scale,
                  void* alibi,
                  void* attn_mask,
                  uint8_t* causal_mask,
                  bool select_nfltmax_at_0,
                  size_t len,
                  size_t total_size,
                  ov::element::Type precision,
                  ov::element::Type attn_mask_prec,
                  ov::element::Type dst_precision);

}  // namespace XARCH
}  // namespace Cpu
}  // namespace Extensions
}  // namespace ov