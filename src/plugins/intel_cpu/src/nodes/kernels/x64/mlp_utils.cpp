// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mlp_utils.hpp"


#include <cstring>
#if defined(HAVE_AVX512F)
#include <immintrin.h>
#endif
#include "../scaled_attn/transpose_kernel.hpp"

namespace ov {
namespace Extensions {
namespace Cpu {
namespace XARCH {

void llm_mlp_transpose_epi32_16x16(void* dst, void* src, int stride) {
    transpose_16x16_kernel(reinterpret_cast<uint32_t*>(dst), reinterpret_cast<uint32_t*>(src), 16, stride/sizeof(uint32_t));
}

}  // namespace XARCH
}  // namespace Cpu
}  // namespace Extensions
}  // namespace ov


