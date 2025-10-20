// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <fstream>
#include <utility>

#if defined(OPENVINO_ARCH_X86_64)
#    include "nodes/kernels/x64/brgemm_kernel.hpp"
#elif defined(OPENVINO_ARCH_ARM64)
#    include "nodes/kernels/aarch64/brgemm_kernel.hpp"
#endif
#include "openvino/core/parallel.hpp"
#include "openvino/core/shape.hpp"
#include "openvino/core/type/element_type.hpp"
#include "utils/plain_tensor.hpp"

#if defined(HAVE_AVX2) || defined(HAVE_AVX512F)
#    include <immintrin.h>

#    include "common.hpp"
#endif
#include "softmax_kernel.hpp"
#include "transpose_kernel.hpp"

using namespace ov::intel_cpu;
namespace ov::Extensions::Cpu::XARCH {

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

struct Xattn {
    std::vector<uint8_t> _scratch_a;
    std::vector<uint8_t> _wsp;
    std::vector<std::shared_ptr<BrgemmKernel>> _xattn_gemm;
    PlainTensor _attn_sum_temp;
    PlainTensor _attn_sum;
    PlainTensor _key_repack;
    size_t _k_num_to_pad = 0;
    size_t _k_num_strided = 0;
    size_t _q_num_strided = 0;
    size_t _q_num_blocks = 0;
    size_t _k_num_blocks = 0;
    size_t _num_per_block = 0;
    size_t _n_block_size = 32;
    size_t _kv_head_groups = 0;
    size_t _m_block_size = 0;

    void sum_blocks_ref(const float* a,
                        size_t M,
                        size_t a_stride,
                        float* dst,
                        size_t out_stride,
                        size_t num_per_block) {
        // TODO: Here we assume the output block is square. So only the row parameter is passed. Need to pass both row
        // and col num to support multi-chunks.
        size_t block_num = div_up(M, num_per_block);
        for (size_t row = 0; row < block_num; row++) {
            for (size_t col = 0; col < block_num; col++) {
                float value = 0.0f;
                for (size_t i = 0; i < num_per_block; i++) {
                    for (size_t j = 0; j < num_per_block; j++) {
                        auto r_idx = row * num_per_block + i;
                        auto c_idx = col * num_per_block + j;
                        if (r_idx < M && c_idx < M) {
                            auto in = a[r_idx * a_stride + c_idx];
                            value += in;
                        }
                    }
                }
                dst[row * out_stride + col] = value;
            }
        }
    }

#if defined(HAVE_AVX512F) || defined(HAVE_AVX2)
    void sum_blocks8x8(const float* a, size_t M, size_t a_stride, float* out, size_t out_stride) {
        size_t block_num = (M + 7) / 8;
        size_t col_num = block_num * 8;  // TODO: Need to pass both row and col num parameter to support multi-chunks
        size_t i = 0;
        for (; i + 8 <= M; i += 8) {
            for (size_t j = 0; j + 8 <= col_num; j += 8) {
                __m256 r0 = _mm256_loadu_ps(a + (i + 0) * a_stride + j);
                __m256 r1 = _mm256_loadu_ps(a + (i + 1) * a_stride + j);
                __m256 r2 = _mm256_loadu_ps(a + (i + 2) * a_stride + j);
                __m256 r3 = _mm256_loadu_ps(a + (i + 3) * a_stride + j);
                __m256 r4 = _mm256_loadu_ps(a + (i + 4) * a_stride + j);
                __m256 r5 = _mm256_loadu_ps(a + (i + 5) * a_stride + j);
                __m256 r6 = _mm256_loadu_ps(a + (i + 6) * a_stride + j);
                __m256 r7 = _mm256_loadu_ps(a + (i + 7) * a_stride + j);

                __m256 sum = _mm256_add_ps(r0, r1);
                sum = _mm256_add_ps(sum, r2);
                sum = _mm256_add_ps(sum, r3);
                sum = _mm256_add_ps(sum, r4);
                sum = _mm256_add_ps(sum, r5);
                sum = _mm256_add_ps(sum, r6);
                sum = _mm256_add_ps(sum, r7);
                hsum(sum);

                const int ib = i >> 3;
                const int jb = j >> 3;
                const float block_sum = _mm256_cvtss_f32(sum);
                out[ib * out_stride + jb] = block_sum;
            }
        }

        auto tails = M - i;
        if (tails) {
            for (size_t j = 0; j + 8 <= col_num; j += 8) {
                __m256 sum = _mm256_setzero_ps();
                for (size_t row = i; row < M; row++) {
                    __m256 r = _mm256_loadu_ps(a + row * a_stride + j);
                    sum = _mm256_add_ps(sum, r);
                }
                hsum(sum);
                const float block_sum = _mm256_cvtss_f32(sum);
                const int jb = j >> 3;
                out[(block_num - 1) * out_stride + jb] = block_sum;
            }
        }
    }
#endif

    void init(size_t B,
              size_t H,
              size_t L,
              size_t S,
              size_t K_H,
              size_t xattn_stride,
              size_t xattn_block_size,
              size_t m_block_size,
              ov::element::Type in_type) {
        _kv_head_groups = H / K_H;
        // The k length should first divided by stride, and the result should be divisible by 32 since block
        // size in brgemm computation is 32. Therefore align k to multiple of xattn_stride * 32.
        auto k_padded = rnd_up(B, xattn_stride * 32);
        _k_num_to_pad = k_padded - B;
        _k_num_strided = div_up(k_padded, xattn_stride);

        _q_num_strided = div_up(B, xattn_stride);
        _q_num_blocks = div_up(B, xattn_block_size);
        _k_num_blocks = div_up(B, xattn_block_size);
        _num_per_block = xattn_block_size / xattn_stride;
        _m_block_size = m_block_size;
        size_t n_num_blocks = _k_num_strided / _n_block_size;

        if (_q_num_blocks <= 1 || _k_num_blocks <= 1 || _num_per_block == 0) {
            return;
        }

        if (_xattn_gemm.empty()) {
            _xattn_gemm.resize(_m_block_size);
            for (size_t i = 1; i < _m_block_size + 1; i++) {
                _xattn_gemm[i - 1] = std::make_shared<BrgemmKernel>(i,                         // M
                                                                    _n_block_size,             // N
                                                                    S,                         // K
                                                                    S * xattn_stride * H * L,  // lda
                                                                    _n_block_size,             // ldb
                                                                    _k_num_strided,            // ldc
                                                                    false,
                                                                    in_type,
                                                                    true);
            }
            auto max_threads_num = parallel_get_max_threads();
            auto scratch_a_size = _xattn_gemm.back()->get_scratch_a_size();
            auto wsp_size = _xattn_gemm.back()->get_wsp_size();

            _scratch_a.assign(scratch_a_size * max_threads_num, 0.0f);
            _wsp.assign(wsp_size * max_threads_num, 0.0f);

            _attn_sum_temp.resize({H, L, _q_num_strided, _k_num_strided}, sizeof(float), ov::element::Type_t::f32);
            ov::parallel_for3d(_q_num_strided, H, L, [&](size_t b, size_t h, size_t l) {
                memset(_attn_sum_temp.ptr<float>(h, l, b, 0), 0, _k_num_strided * sizeof(float));
            });

            _attn_sum.resize({H, L, _q_num_blocks, _k_num_blocks}, _attn_sum_temp.m_element_size, _attn_sum_temp.m_dt);

            _key_repack.resize({K_H, L, S * n_num_blocks, _n_block_size}, sizeof(float), ov::element::Type_t::f32);
        }
    }

    void estimate(const PlainTensor& query,
                  const PlainTensor& key,
                  size_t block_size,
                  size_t stride,
                  float threshold,
                  PlainTensor& mask) {
        auto B = query.size(0);
        auto H = query.size(1);
        auto L = query.size(2);
        auto S = query.size(3);
        auto K_H = key.size(1);
        OPENVINO_ASSERT(query.size(0) == key.size(0));

        const auto m_num_blocks = div_up(_q_num_strided, _m_block_size);
        auto in_type = query.m_dt;
        auto scratch_a_size = _xattn_gemm.back()->get_scratch_a_size();
        auto wsp_size = _xattn_gemm.back()->get_wsp_size();
        const auto n_num_blocks = _k_num_strided / _n_block_size;

        for (size_t i = 0; i < stride; i++) {
            // repack key buffer
            ov::parallel_for2d(K_H, n_num_blocks, [&](size_t h, size_t n_blk) {
                auto n_start = n_blk * _n_block_size;
                size_t need_to_pad = (_k_num_to_pad + i) / stride;
                size_t N = (n_blk == n_num_blocks - 1) ? _n_block_size - need_to_pad : 32;

                if (N) {
                    void* src = key.ptr_v(n_start * stride + i, h, 0, 0);
                    void* dst = _key_repack.ptr_v(h, 0, S * n_blk, 0);
                    if (in_type == ov::element::bf16) {
#if defined(HAVE_AVX512F)
                        transpose_16NxK<bfloat16, ov::element::bf16>(reinterpret_cast<bfloat16*>(dst),  // dst
                                                                     reinterpret_cast<bfloat16*>(src),  // src
                                                                     N,                                 // N
                                                                     S,                                 // K
                                                                     _n_block_size,                     // block size
                                                                     _n_block_size,                     // dst stride
                                                                     (S * stride * K_H)                 // src stride
                        );
#else
                    OPENVINO_THROW("xattention: bf16 needs avx512+ hardware.");
#endif
                    } else if (in_type == ov::element::f32) {
                        transpose_16NxK<float, ov::element::f32>(reinterpret_cast<float*>(dst),  // dst
                                                                 src,                            // src
                                                                 N,                              // N
                                                                 S,                              // K
                                                                 _n_block_size,                  // block size
                                                                 _n_block_size,                  // dst stride
                                                                 (S * stride * K_H)              // src stride
                        );
                    } else {
                        OPENVINO_THROW("xattention: unsupported precision: ", in_type);
                    }

                } else {
                    for (size_t k = 0; k < S; k++) {
                        auto* dst = _key_repack.ptr_v(h, 0, S * n_blk + k, 0);
                        memset(dst, 0, _n_block_size * _key_repack.m_element_size);
                    }
                }
            });

            ov::parallel_for2d(H, m_num_blocks, [&](size_t h, size_t m_blk) {
                auto ithr = parallel_get_thread_num();
                auto m_start = m_blk * _m_block_size;
                auto m_end = std::min(m_start + _m_block_size, _q_num_strided);
                auto m_cnt = m_end - m_start;

                auto q_index = m_start * stride + stride - i - 1;
                auto q_end = q_index + stride * (m_cnt - 1);

                // q_end may extend the seq length since it was aligned to multiple of stride
                if (q_end >= B) {
                    m_cnt--;
                }

                if (m_cnt > 0) {
                    auto* q_ptr = query.ptr_v(stride - 1 - i + m_start * stride, h, 0, 0);
                    auto cur_k_block_num = m_blk + 1;
                    for (size_t n_blk = 0; n_blk < cur_k_block_num; n_blk++) {
                        auto n_start = n_blk * S;
                        auto* k_ptr = _key_repack.ptr_v(h / _kv_head_groups, 0, n_start, 0);
                        auto* c_ptr = &_attn_sum_temp.at<float>({h, 0, m_start, n_blk * _n_block_size});
                        _xattn_gemm[m_cnt - 1]->executeGemm(m_cnt < _m_block_size,
                                                            q_ptr,
                                                            k_ptr,
                                                            c_ptr,
                                                            nullptr,
                                                            nullptr,
                                                            _wsp.data() + wsp_size * ithr,
                                                            _scratch_a.data() + scratch_a_size * ithr);
                    }
                }
            });
        }

        parallel_for3d(_q_num_strided, H, L, [&](size_t b, size_t h, size_t l) {
            auto* data = _attn_sum_temp.ptr<float>(h, l, b, 0);
            auto ncausal = b + 1;
            auto scale = 1.0 / sqrt(S) / stride;
            attn_softmax_kernel<float>(data,
                                       reinterpret_cast<float*>(data),
                                       scale,
                                       nullptr,
                                       nullptr,
                                       nullptr,
                                       false,
                                       ncausal,
                                       _k_num_strided,
                                       ov::element::f32,
                                       ov::element::f32,
                                       0);
        });

        size_t src_stride = _attn_sum_temp.size(3);
        size_t dst_stride = _attn_sum.size(3);
        size_t row_num = _attn_sum_temp.size(2);
        parallel_for2d(H, L, [&](size_t h, size_t l) {
            auto* src = _attn_sum_temp.ptr<float>(h, l, 0, 0);
            auto* dst = _attn_sum.ptr<float>(h, l, 0, 0);
#if defined(HAVE_AVX512F) || defined(HAVE_AVX2)
            if (_num_per_block == 8) {
                sum_blocks8x8(src, row_num, src_stride, dst, dst_stride);
            } else {
                sum_blocks_ref(src, row_num, src_stride, dst, dst_stride, _num_per_block);
            }
#else
        sum_blocks_ref(src, row_num, src_stride, dst, dst_stride, _num_per_block);
#endif
        });

        // Find blocks
        mask.resize({H, _q_num_blocks, _k_num_blocks}, sizeof(bool), ov::element::Type_t::boolean);
        parallel_for3d(_q_num_blocks, H, L, [&](size_t q_n, size_t h, size_t l) {
            auto* row = _attn_sum.ptr<float>(h, l, q_n, 0);
            float required_sum = std::accumulate(row, row + _k_num_blocks, 0.0f, std::plus<>()) * threshold;

            std::vector<std::pair<float, int>> values_with_index(_k_num_blocks);
            for (size_t k = 0; k < _k_num_blocks; k++) {
                values_with_index[k] = std::make_pair(row[k], k);
            }

            // The blocks in the first column and along the main diagonal should always be selected thus excluded from
            // the sorting process.
            if (q_n > 1) {
                std::swap(values_with_index[1], values_with_index[q_n]);
            }
            std::sort(values_with_index.begin() + 2,
                      values_with_index.end(),
                      [](const std::pair<float, int>& a, const std::pair<float, int>& b) {
                          return a.first > b.first;
                      });

            // Compute the cumulative sum. Set the values in the first and second columns to zero to ensures that the
            // blocks in the first column and along the main diagonal are always selected.
            std::vector<float> cumsum_without_self(_k_num_blocks, 0.0f);
            for (size_t i = 1; i < _k_num_blocks; i++) {
                cumsum_without_self[i] = values_with_index[i - 1].first + cumsum_without_self[i - 1];
            }
            cumsum_without_self[1] = 0.0f;

            for (size_t i = 0; i < _k_num_blocks; i++) {
                bool value = (cumsum_without_self[i] < required_sum);
                if (i > q_n) {
                    value = false;
                }
                *(mask.ptr<bool>(h, q_n, values_with_index[i].second)) = value;
            }
        });
    }
};

}  // namespace ov::Extensions::Cpu::XARCH
