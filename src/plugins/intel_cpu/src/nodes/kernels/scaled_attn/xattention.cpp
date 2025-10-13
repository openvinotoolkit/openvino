#include "xattention.hpp"

#include <algorithm>
#include <chrono>
#include <common/z_magic.hpp>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <ios>
#include <limits>
#include <numeric>
#include <vector>

#include "nodes/kernels/x64/brgemm_kernel.hpp"
#include "openvino/core/parallel.hpp"
#include "openvino/core/type/element_type.hpp"

#if defined(HAVE_AVX2) || defined(HAVE_AVX512F)
#    include <immintrin.h>

#    include "common.hpp"
#endif
#include "softmax_kernel.hpp"
#include "transpose_kernel.hpp"

using namespace ov::intel_cpu;
namespace ov::Extensions::Cpu::XARCH {

namespace ref {
void softmax(float* a, int len) {
    float max = *std::max_element(a, a + len);
    float sum = 0.0F;
    for (int i = 0; i < len; i++) {
        a[i] = ::exp(a[i] - max);
        sum += a[i];
    }
    float scale = 1.0F / sum;
    for (int i = 0; i < len; i++) {
        a[i] *= scale;
    }
}

template <typename D>
float dot_product(const D* a, const D* b, int len, int stride_b = 1) {
    float result = 0;
    if (stride_b == 1) {
        for (int i = 0; i < len; i++) {
            result += static_cast<float>(a[i]) * static_cast<float>(b[i]);
        }
    } else {
        for (int i = 0; i < len; i++) {
            result += static_cast<float>(a[i]) * static_cast<float>(b[i * stride_b]);
        }
    }
    return result;
}

void sum_blocks_ref(const float* a, size_t M, size_t a_stride, float* dst, size_t out_stride, size_t num_per_block) {
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

}  // namespace ref

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

#if defined(HAVE_AVX512F) || defined(HAVE_AVX2)
void sum_blocks8x8(const float* a, size_t M, size_t a_stride, float* out, size_t out_stride) {
    size_t block_num = (M + 7) / 8;
    size_t col_num = block_num * 8;
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

PlainTensor xattn_estimate(const PlainTensor& query,
                           const PlainTensor& key,
                           size_t block_size,
                           size_t stride,
                           float threshold) {
    auto B = query.size(0);
    auto H = query.size(1);
    auto L = query.size(2);
    auto S = query.size(3);
    OPENVINO_ASSERT(query.size(0) == key.size(0));

    const auto K_H = key.size(1);
    const auto kv_head_groups = H / K_H;

    // Align k length to multiple of stride and multiple of 32, since block
    // size in brgemm computation is 32.
    auto k_padded = rnd_up(B, stride * 32);
    auto k_num_to_pad = k_padded - B;
    auto k_num_strided = div_up(k_padded, stride);
    auto q_num_strided = div_up(B, stride);
    auto q_num_blocks = div_up(B, block_size);
    auto k_num_blocks = div_up(B, block_size);
    auto num_per_block = block_size / stride;

    if (q_num_blocks <= 1 || k_num_blocks <= 1 || num_per_block == 0) {
        return {};
    }

    PlainTensor attn_sum_temp;
    attn_sum_temp.resize({H, L, q_num_strided, k_num_strided}, sizeof(float), ov::element::Type_t::f32);
    parallel_for3d(q_num_strided, H, L, [&](size_t b, size_t h, size_t l) {
        memset(attn_sum_temp.ptr<float>(h, l, b, 0), 0, k_num_strided * sizeof(float));
    });

    const size_t n_block_size = 32;
    const auto m_block_size = BrgemmKernel::get_mblk_size();
    const auto m_num_blocks = div_up(q_num_strided, m_block_size);
    auto in_type = query.m_dt;

    std::vector<std::shared_ptr<BrgemmKernel>> kernels(m_block_size);
    for (size_t i = 1; i < m_block_size + 1; i++) {
        kernels[i - 1] = std::make_shared<BrgemmKernel>(i,                   // M
                                                        n_block_size,        // N
                                                        S,                   // K
                                                        S * stride * H * L,  // lda
                                                        n_block_size,        // ldb
                                                        k_num_strided,       // ldc
                                                        false,
                                                        in_type,
                                                        true);
    }

    auto max_threads_num = parallel_get_max_threads();
    auto scratch_a_size = kernels.back()->get_scratch_a_size();
    auto scratch_b_size = kernels.back()->get_scratch_b_size();
    auto wsp_size = kernels.back()->get_wsp_size();

    std::vector<uint8_t> scratch_a(scratch_a_size * max_threads_num, 0.0f);
    std::vector<uint8_t> scratch_b(scratch_b_size * max_threads_num, 0.0f);
    std::vector<uint8_t> wsp(wsp_size * max_threads_num, 0.0f);
    const auto n_num_blocks = k_num_strided / n_block_size;

    // Repack key buffer to [H, L, S * n_num_block, n_block_size]
    PlainTensor key_repack;
    key_repack.resize({K_H, L, S * n_num_blocks, n_block_size}, sizeof(float), ov::element::Type_t::f32);

    for (size_t i = 0; i < stride; i++) {
        ov::parallel_for2d(K_H, n_num_blocks, [&](size_t h, size_t n_blk) {
            auto n_start = n_blk * n_block_size;
            size_t need_to_pad = (k_num_to_pad + i) / stride;
            size_t N = (n_blk == n_num_blocks - 1) ? n_block_size - need_to_pad : 32;

            if (N) {
                void* src = key.ptr_v(n_start * stride + i, h, 0, 0);
                void* dst = key_repack.ptr_v(h, 0, S * n_blk, 0);
                if (in_type == ov::element::bf16) {
#if defined(HAVE_AVX512F)
                    transpose_16NxK<bfloat16, ov::element::bf16>(reinterpret_cast<bfloat16*>(dst),  // dst
                                                                 reinterpret_cast<bfloat16*>(src),  // src
                                                                 N,                                 // N
                                                                 S,                                 // K
                                                                 n_block_size,                      // block size
                                                                 n_block_size,                      // dst stride
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
                                                             n_block_size,                   // block size
                                                             n_block_size,                   // dst stride
                                                             (S * stride * K_H)              // src stride
                    );
                } else {
                    OPENVINO_THROW("xattention: unsupported precision: ", in_type);
                }

            } else {
                for (size_t k = 0; k < S; k++) {
                    auto* dst = key_repack.ptr_v(h, 0, S * n_blk + k, 0);
                    memset(dst, 0, n_block_size * key_repack.m_element_size);
                }
            }
        });

        ov::parallel_for2d(H, m_num_blocks, [&](size_t h, size_t m_blk) {
            auto ithr = parallel_get_thread_num();
            auto m_start = m_blk * m_block_size;
            auto m_end = std::min(m_start + m_block_size, q_num_strided);
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
                    auto* k_ptr = key_repack.ptr_v(h / kv_head_groups, 0, n_start, 0);
                    auto* c_ptr = &attn_sum_temp.at<float>({h, 0, m_start, n_blk * n_block_size});
                    kernels[m_cnt - 1]->executeGemm(m_cnt < m_block_size,
                                                    q_ptr,
                                                    k_ptr,
                                                    c_ptr,
                                                    nullptr,
                                                    nullptr,
                                                    wsp.data() + wsp_size * ithr,
                                                    scratch_a.data() + scratch_a_size * ithr);
                }
            }
        });
    }

    parallel_for3d(q_num_strided, H, L, [&](size_t b, size_t h, size_t l) {
        auto* data = attn_sum_temp.ptr<float>(h, l, b, 0);
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
                                   k_num_strided,
                                   ov::element::f32,
                                   ov::element::f32,
                                   0);
    });

    PlainTensor attn_sum;
    attn_sum.resize({H, L, q_num_blocks, k_num_blocks}, attn_sum_temp.m_element_size, attn_sum_temp.m_dt);
    size_t src_stride = attn_sum_temp.size(3);
    size_t dst_stride = attn_sum.size(3);
    size_t row_num = attn_sum_temp.size(2);
    parallel_for2d(H, L, [&](size_t h, size_t l) {
        auto* src = attn_sum_temp.ptr<float>(h, l, 0, 0);
        auto* dst = attn_sum.ptr<float>(h, l, 0, 0);
#if defined(HAVE_AVX512F) || defined(HAVE_AVX2)
        if (num_per_block == 8) {
            sum_blocks8x8(src, row_num, src_stride, dst, dst_stride);
        } else {
            ref::sum_blocks_ref(src, row_num, src_stride, dst, dst_stride, num_per_block);
        }
#else
        ref::sum_blocks_ref(src, row_num, src_stride, dst, dst_stride, num_per_block);
#endif
    });

    // Find blocks
    PlainTensor mask;
    mask.resize({H, q_num_blocks, k_num_blocks}, sizeof(bool), ov::element::Type_t::boolean);
    parallel_for3d(q_num_blocks, H, L, [&](size_t q_n, size_t h, size_t l) {
        auto* row = attn_sum.ptr<float>(h, l, q_n, 0);
        float required_sum = std::accumulate(row, row + k_num_blocks, 0.0f, std::plus<>()) * threshold;

        std::vector<std::pair<float, int>> values_with_index(k_num_blocks);
        for (size_t k = 0; k < k_num_blocks; k++) {
            values_with_index[k] = std::make_pair(row[k], k);
        }

        // The blocks in the first column and along the main diagonal should always be selected thus excluded from the
        // sorting process.
        if (q_n > 1) {
            std::swap(values_with_index[1], values_with_index[q_n]);
        }
        std::sort(values_with_index.begin() + 2,
                  values_with_index.end(),
                  [](const std::pair<float, int>& a, const std::pair<float, int>& b) {
                      return a.first > b.first;
                  });

        // Compute the cumulative sum. Set the values in the first and second columns to zero to ensures that the blocks
        // in the first column and along the main diagonal are always selected.
        std::vector<float> cumsum_without_self(k_num_blocks, 0.0f);
        for (size_t i = 1; i < k_num_blocks; i++) {
            cumsum_without_self[i] = values_with_index[i - 1].first + cumsum_without_self[i - 1];
        }
        cumsum_without_self[1] = 0.0f;

        for (size_t i = 0; i < k_num_blocks; i++) {
            bool value = (cumsum_without_self[i] < required_sum);
            if (i > q_n) {
                value = false;
            }
            *(mask.ptr<bool>(h, q_n, values_with_index[i].second)) = value;
        }
    });

    return mask;
}
}  // namespace ov::Extensions::Cpu::XARCH
