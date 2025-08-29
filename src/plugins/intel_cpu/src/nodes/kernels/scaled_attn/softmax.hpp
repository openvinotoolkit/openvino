// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <cstddef>
#include <cstdint>
#include <openvino/core/type/element_type.hpp>

namespace ov::Extensions::Cpu::XARCH {

void attn_softmax(void* a,
                  void* a_dst,
                  float scale,
                  void* alibi,
                  void* attn_mask,
                  uint8_t* causal_mask,
                  bool select_nfltmax_at_0,
                  size_t seq_len,
                  size_t kv_total_len,
                  ov::element::Type precision,
                  ov::element::Type attn_mask_prec,
                  ov::element::Type dst_precision);

}  // namespace ov::Extensions::Cpu::XARCH
