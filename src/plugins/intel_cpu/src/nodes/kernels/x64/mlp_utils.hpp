// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <cstddef>
#include <vector>
#include <array>

namespace ov {
namespace Extensions {
namespace Cpu {
namespace XARCH {

void llm_mlp_transpose_epi32_16x16(void* dst, void* src, int stride);

}  // namespace XARCH
}  // namespace Cpu
}  // namespace Extensions
}  // namespace ov
