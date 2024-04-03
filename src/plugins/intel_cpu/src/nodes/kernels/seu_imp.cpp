// Copyright (C) 2018-2024 Intel Corporation
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

// #include "openvino/core/type/bfloat16.hpp"
// #include "openvino/core/parallel.hpp"
// #include "scaled_attn/common.hpp"
#include "seu_imp.hpp"

namespace ov {
namespace Extensions {
namespace Cpu {
namespace XARCH {

// avx512/avx2 register length in byte
static constexpr size_t vec_len_avx512 = 64lu;
static constexpr size_t vec_len_avx2 = 32lu;
// avx512/avx2 register length in float
static constexpr size_t vec_len_f32_avx512 = vec_len_avx512 / sizeof(float);
static constexpr size_t vec_len_f32_avx2 = vec_len_avx2 / sizeof(float);

void reduce_add(char* data_in_bytes, const size_t data_stride,
                const char* updates_in_bytes, const size_t update_stride, const size_t size) {
    size_t i = 0;
    float* data = reinterpret_cast<float*>(data_in_bytes);
    float* updates = const_cast<float*>(reinterpret_cast<const float*>(updates_in_bytes)); 
#if defined(HAVE_AVX512F)
    // process vector body
    while (i + vec_len_f32_avx512 <= size) {
        auto v_a = _mm512_loadu_ps(data);
        auto v_b = _mm512_loadu_ps(updates);
        v_a = _mm512_add_ps(v_a, v_b);
        _mm512_storeu_ps(data, v_a);

        data += vec_len_f32_avx512;
        updates += vec_len_f32_avx512;
        i += vec_len_f32_avx512;
    }
#elif defined(HAVE_AVX2)
    //
    tt
#endif
    // process tail
    // std::cout << "======" << i << ", " << size << std::endl;
    for (; i < size; i++) {
        *data += *updates;
        data++;
        updates++;
    }
}

}  // namespace XARCH
}  // namespace Cpu
}  // namespace Extensions
}  // namespace ov
