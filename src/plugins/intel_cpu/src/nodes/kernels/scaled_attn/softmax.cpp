// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include <cfloat>
#include <cmath>
#include <cstdint>
#include <cstring>

#include "openvino/core/type/element_type.hpp"

#if defined(HAVE_AVX2) || defined(HAVE_AVX512F)
#    include <immintrin.h>
#endif

#include "softmax.hpp"
#include "softmax_kernel.hpp"

namespace ov::Extensions::Cpu::XARCH {

void attn_softmax(void* a,
                  void* a_dst,
                  float scale,
                  void* alibi,
                  void* attn_mask,
                  uint8_t* causal_mask,
                  bool select_nfltmax_at_0,
                  size_t len,
                  size_t total_size,
                  [[maybe_unused]] ov::element::Type precision,
                  ov::element::Type attn_mask_prec,
                  ov::element::Type dst_precision) {
#if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
    if (precision == ov::element::f16) {
        auto _a = reinterpret_cast<ov::float16*>(a);
        auto _alibi = reinterpret_cast<ov::float16*>(alibi);
        attn_softmax_kernel<ov::float16>(_a,
                                         a_dst,
                                         scale,
                                         _alibi,
                                         attn_mask,
                                         causal_mask,
                                         select_nfltmax_at_0,
                                         len,
                                         total_size,
                                         attn_mask_prec,
                                         dst_precision);
        return;
    }
#endif
    auto* _a = reinterpret_cast<float*>(a);
    auto* _alibi = reinterpret_cast<float*>(alibi);
    attn_softmax_kernel<float>(_a,
                               a_dst,
                               scale,
                               _alibi,
                               attn_mask,
                               causal_mask,
                               select_nfltmax_at_0,
                               len,
                               total_size,
                               attn_mask_prec,
                               dst_precision);
}

}  // namespace ov::Extensions::Cpu::XARCH
