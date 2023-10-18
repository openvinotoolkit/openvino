// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace InferenceEngine {
namespace Extensions {
namespace Cpu {
namespace XARCH {

void attn_softmax(float* a,
                  float scale,
                  float* alibi,
                  float* attn_mask,
                  uint8_t* causal_mask,
                  bool select_nfltmax_at_0,
                  size_t len,
                  size_t total_size);
}  // namespace XARCH
}  // namespace Cpu
}  // namespace Extensions
}  // namespace InferenceEngine