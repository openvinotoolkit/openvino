#pragma once

#include <type_traits>

#include "openvino/core/type/element_type.hpp"
#include "transpose_kernel.hpp"
#include "utils/general_utils.h"

namespace ov::Extensions::Cpu::XARCH {
using namespace ov::intel_cpu;

template <typename TDST, typename TSRC>
inline void transpose_tailx16_kernel(TDST* dst,
                                     TSRC* src,
                                     size_t n_cnt,
                                     size_t k_cnt,
                                     size_t dst_stride,
                                     size_t src_stride) {
    for (size_t i = 0; i < n_cnt; i++) {
        for (size_t j = 0; j < k_cnt; j++) {
            dst[j * dst_stride + i] = static_cast<TDST>(src[j + i * src_stride]);
        }
    }
}

template <typename TDST,
          ov::element::Type_t SRC_PREC,
          std::enable_if_t<(none_of(SRC_PREC, ov::element::i8, ov::element::u8, ov::element::u4)), bool> = true>
void transpose_16NxK(TDST* dst,
                     void* src,
                     const size_t N,
                     const size_t K,
                     const size_t block_size,
                     const size_t dst_stride,
                     const size_t src_stride) {
    size_t k = 0;
    auto* src_ptr = reinterpret_cast<typename ov::element_type_traits<SRC_PREC>::value_type*>(src);
    for (size_t k = 0; k < K; k++) {
        memset(dst + k * dst_stride + N, 0, (block_size - N) * sizeof(TDST));
    }

    for (; k + 16 <= K; k += 16) {
        size_t n = 0;
        for (; n + 16 <= N; n += 16) {
            transpose_16x16_kernel(dst + n, src_ptr + n * src_stride, dst_stride, src_stride);
        }

        if (n < block_size) {
            transpose_tailx16_kernel(dst + n, src_ptr + n * src_stride, N - n, 16, dst_stride, src_stride);
        }

        dst += 16 * dst_stride;
        src_ptr += 16;
    }
    if (k < K) {
        size_t n = 0;
        for (; n + 16 <= N; n += 16) {
            transpose_16xK_kernel(dst + n, src_ptr + n * src_stride, K - k, dst_stride, src_stride);
        }

        if (n < block_size) {
            transpose_tailx16_kernel(dst + n, src_ptr + n * src_stride, N - n, K - k, dst_stride, src_stride);
        }
    }
}

#if defined(HAVE_AVX512F)
template <typename T,
          ov::element::Type_t SRC_PREC,
          typename std::enable_if<(SRC_PREC == ov::element::bf16 || SRC_PREC == ov::element::f16) &&
                                      (SRC_PREC == precision_of<T>::value),
                                  bool>::type = true>
static void transpose_16NxK(T* dst,
                            T* src,
                            const size_t N,
                            const size_t K,
                            const size_t block_size,
                            const size_t dst_stride,
                            const size_t src_stride) {
    // will treat as uint32_t transpose
    auto s = reinterpret_cast<uint32_t*>(src);
    auto d = reinterpret_cast<uint32_t*>(dst);
    transpose_16NxK<uint32_t, ov::element::u32>(d, s, N, K >> 1, block_size, dst_stride, src_stride >> 1);
}
#endif
}  // namespace ov::Extensions::Cpu::XARCH